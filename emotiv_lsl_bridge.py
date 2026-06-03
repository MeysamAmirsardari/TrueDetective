"""
Emotiv Cortex → LSL Bridge
==========================
Connects to the Emotiv Cortex service over its local WebSocket JSON-RPC
endpoint, authenticates, opens a session with the first available headset,
subscribes to the raw EEG stream, and rebroadcasts every sample over Lab
Streaming Layer (LSL) for downstream consumers (PsychoPy, BCILab, etc.).

Requirements:
    pip install websockets pylsl

Before running:
    1.  Install and launch the Emotiv Launcher / Cortex service.
    2.  Register a Cortex application at
        https://www.emotiv.com/my-account/cortex-apps/  to obtain
        a CLIENT_ID and CLIENT_SECRET. Paste them below.
    3.  Raw EEG access requires an active EEG-tier license on your
        Emotiv account. Without it, the `subscribe` call will return
        a permission error (handled & explained below).
    4.  Power on your headset (Insight, EPOC, EPOC+, EPOC X, MN8, …)
        and pair it through the Launcher.

Usage:
    python emotiv_lsl_bridge.py
"""

import asyncio
import json
import logging
import ssl
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import websockets
from pylsl import StreamInfo, StreamOutlet

# =============================================================================
# CONFIGURATION
# =============================================================================

CLIENT_ID = "YOUR_CORTEX_CLIENT_ID"
CLIENT_SECRET = "YOUR_CORTEX_CLIENT_SECRET"
LICENSE = ""                # leave "" to use the account default license
DEBIT = 1                   # session-debit count for license metering

CORTEX_URL = "wss://localhost:6868"

# Per-headset nominal sampling rates and channel labels.
# (Cortex still tells us the live layout via the `subscribe` response;
#  this dict is only used for the LSL nominal_srate and a friendly fallback.)
HEADSET_PROFILES: Dict[str, Dict[str, Any]] = {
    "INSIGHT":   {"srate": 128.0, "channels": ["AF3", "T7", "Pz", "T8", "AF4"]},
    "INSIGHT2":  {"srate": 128.0, "channels": ["AF3", "T7", "Pz", "T8", "AF4"]},
    "EPOC":      {"srate": 128.0, "channels": ["AF3", "F7", "F3", "FC5", "T7",
                                               "P7", "O1", "O2", "P8", "T8",
                                               "FC6", "F4", "F8", "AF4"]},
    "EPOCPLUS":  {"srate": 128.0, "channels": ["AF3", "F7", "F3", "FC5", "T7",
                                               "P7", "O1", "O2", "P8", "T8",
                                               "FC6", "F4", "F8", "AF4"]},
    "EPOCX":     {"srate": 128.0, "channels": ["AF3", "F7", "F3", "FC5", "T7",
                                               "P7", "O1", "O2", "P8", "T8",
                                               "FC6", "F4", "F8", "AF4"]},
    "EPOCFLEX":  {"srate": 128.0, "channels": []},   # user-defined montage
    "MN8":       {"srate": 128.0, "channels": ["T7", "T8"]},
}

# Cortex error codes worth special-casing.
ERR_LICENSE_NO_EEG_ACCESS = -32046     # "Your license does not allow raw EEG"
ERR_USER_NOT_LOGGED_IN = -32018

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("emotiv-lsl")


# =============================================================================
# Errors
# =============================================================================

class CortexError(Exception):
    """Wraps a JSON-RPC error returned by the Cortex service."""
    def __init__(self, code: int, message: str):
        super().__init__(f"[Cortex {code}] {message}")
        self.code = code
        self.message = message


# =============================================================================
# Cortex client (JSON-RPC 2.0 over WSS)
# =============================================================================

@dataclass
class CortexClient:
    client_id: str
    client_secret: str
    license: str = ""
    debit: int = 1
    url: str = CORTEX_URL

    ws: Optional[websockets.WebSocketClientProtocol] = None
    cortex_token: Optional[str] = None
    session_id: Optional[str] = None
    headset_id: Optional[str] = None
    headset_type: Optional[str] = None
    eeg_columns: List[str] = field(default_factory=list)

    _msg_id: int = 0
    _pending: Dict[int, "asyncio.Future[Any]"] = field(default_factory=dict)
    _stream_queue: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)
    _reader_task: Optional[asyncio.Task] = None

    # ── Connection lifecycle ────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the WSS connection. Cortex uses a self-signed local cert."""
        log.info("Connecting to Cortex at %s", self.url)
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        self.ws = await websockets.connect(
            self.url, ssl=ssl_ctx, max_size=2 ** 22, ping_interval=30,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        log.info("WebSocket connection established.")

    async def close(self) -> None:
        if self.session_id:
            try:
                await self._call("updateSession", {
                    "cortexToken": self.cortex_token,
                    "session": self.session_id,
                    "status": "close",
                })
            except Exception as e:
                log.debug("updateSession close failed: %s", e)
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()
        log.info("Connection closed.")

    # ── Internal JSON-RPC plumbing ──────────────────────────────────────

    async def _reader_loop(self) -> None:
        """
        Single coroutine reading every WS frame. JSON-RPC responses with an
        `id` get routed back to the matching pending future; everything else
        (streaming samples, warnings) is queued for the consumer.
        """
        try:
            assert self.ws is not None
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg and msg["id"] in self._pending:
                    fut = self._pending.pop(msg["id"])
                    if not fut.done():
                        fut.set_result(msg)
                else:
                    await self._stream_queue.put(msg)
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed as e:
            log.warning("WebSocket closed: %s", e)
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(e)

    async def _call(self, method: str, params: Dict[str, Any]) -> Any:
        """
        Send a JSON-RPC 2.0 request and await its response.

            { "jsonrpc": "2.0",
              "id":      <unique int>,
              "method":  "<cortex method>",
              "params":  { ... } }
        """
        assert self.ws is not None, "WebSocket not connected."
        self._msg_id += 1
        msg_id = self._msg_id
        payload = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params,
        }
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        await self.ws.send(json.dumps(payload))

        response = await fut
        if "error" in response:
            err = response["error"]
            raise CortexError(err.get("code", 0), err.get("message", "?"))
        return response.get("result")

    # ── Auth flow ───────────────────────────────────────────────────────

    async def authenticate(self) -> str:
        """
        Run the full Cortex auth handshake:
            getUserLogin    → confirm a user is signed into the Launcher
            requestAccess   → ensure THIS client_id has been approved
                              (the user will see a popup in the Launcher
                              the first time this runs)
            authorize       → exchange client credentials for a cortexToken
        """
        # 1. getUserLogin — verify a user is signed into the Launcher.
        login_info = await self._call("getUserLogin", {})
        if not login_info:
            raise CortexError(
                ERR_USER_NOT_LOGGED_IN,
                "No user is signed into the Emotiv Launcher.",
            )
        log.info("Logged-in Emotiv user: %s",
                 login_info[0].get("username", "<unknown>"))

        # 2. requestAccess — must be approved once per CLIENT_ID, per machine.
        access = await self._call("requestAccess", {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
        })
        if not access.get("accessGranted"):
            log.warning("Approval pending — open the Emotiv Launcher and "
                        "approve this application, then re-run.")
            log.warning("Launcher message: %s", access.get("message"))
            raise CortexError(0, "Access not yet granted by user.")

        # 3. authorize — obtain a cortexToken used for every subsequent call.
        result = await self._call("authorize", {
            "clientId": self.client_id,
            "clientSecret": self.client_secret,
            "license": self.license,
            "debit": self.debit,
        })
        self.cortex_token = result["cortexToken"]
        log.info("Authorized. Cortex token acquired.")
        return self.cortex_token

    # ── Headset & session ───────────────────────────────────────────────

    async def find_headset(self) -> str:
        """queryHeadsets → pick the first one that's connected (or any)."""
        headsets = await self._call("queryHeadsets", {})
        if not headsets:
            raise CortexError(0, "No headsets discovered. Is one powered on?")

        # Prefer an already-connected headset; otherwise take the first.
        connected = [h for h in headsets if h.get("status") == "connected"]
        chosen = connected[0] if connected else headsets[0]

        self.headset_id = chosen["id"]
        # Headset id looks like "INSIGHT-A1B2C3D4" — first token is the type.
        self.headset_type = self.headset_id.split("-")[0].upper()

        # If the headset is merely discovered (not connected), bring it up.
        if chosen.get("status") != "connected":
            log.info("Headset %s is %s; issuing controlDevice connect…",
                     self.headset_id, chosen.get("status"))
            await self._call("controlDevice", {
                "command": "connect",
                "headset": self.headset_id,
            })

        log.info("Using headset: %s  (type=%s)",
                 self.headset_id, self.headset_type)
        return self.headset_id

    async def create_session(self) -> str:
        """createSession in 'active' status — required to subscribe to data."""
        assert self.cortex_token and self.headset_id
        result = await self._call("createSession", {
            "cortexToken": self.cortex_token,
            "headset": self.headset_id,
            "status": "active",
        })
        self.session_id = result["id"]
        log.info("Session created: %s", self.session_id)
        return self.session_id

    async def subscribe_eeg(self) -> List[str]:
        """
        Subscribe to the raw 'eeg' stream and learn the column layout.

        The success response looks like:
            { "success": [{ "streamName": "eeg",
                            "cols": ["COUNTER","INTERPOLATED",
                                     "AF3","T7","Pz","T8","AF4",
                                     "RAW_CQ","MARKER_HARDWARE","MARKER"],
                            "sid": "..." }],
              "failure": [] }

        We remember the column layout so we can pick out the real EEG
        channels (everything that isn't a counter / quality / marker).
        """
        assert self.cortex_token and self.session_id
        try:
            result = await self._call("subscribe", {
                "cortexToken": self.cortex_token,
                "session": self.session_id,
                "streams": ["eeg"],
            })
        except CortexError as e:
            if e.code == ERR_LICENSE_NO_EEG_ACCESS:
                log.error(
                    "Your Emotiv license does NOT allow raw EEG access.\n"
                    "       Upgrade to an EEG-tier license at "
                    "https://www.emotiv.com/pricing-plans/  "
                    "and try again."
                )
            raise

        failures = result.get("failure", [])
        if failures:
            raise CortexError(0, f"subscribe failed: {failures}")

        cols = result["success"][0]["cols"]
        self.eeg_columns = cols
        log.info("Subscribed to EEG stream. Columns: %s", cols)
        return cols

    # ── Streaming ───────────────────────────────────────────────────────

    async def stream(self):
        """Async generator yielding every non-response message."""
        while True:
            yield await self._stream_queue.get()


# =============================================================================
# LSL bridge
# =============================================================================

NON_EEG_COLS = {"COUNTER", "INTERPOLATED", "RAW_CQ",
                "MARKER", "MARKER_HARDWARE", "TIME_STAMP"}


def build_lsl_outlet(headset_type: str, eeg_cols: List[str]) -> tuple[StreamOutlet, List[int]]:
    """
    Construct an LSL StreamOutlet whose channels match the *actual* EEG
    columns reported by Cortex (filtering out counters / markers).

    Returns the outlet plus the indices into the Cortex sample array that
    correspond to those EEG channels — used for fast per-sample slicing.
    """
    eeg_indices = [i for i, name in enumerate(eeg_cols)
                   if name.upper() not in NON_EEG_COLS]
    channel_labels = [eeg_cols[i] for i in eeg_indices]

    profile = HEADSET_PROFILES.get(headset_type, {"srate": 128.0})
    srate = profile["srate"]

    info = StreamInfo(
        name=f"Emotiv_{headset_type}_EEG",
        type="EEG",
        channel_count=len(channel_labels),
        nominal_srate=srate,
        channel_format="float32",
        source_id=f"emotiv_{headset_type.lower()}_bridge",
    )

    # Per-channel metadata (LSL XDF consumers read this).
    chans = info.desc().append_child("channels")
    for label in channel_labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")

    info.desc().append_child_value("manufacturer", "Emotiv")
    info.desc().append_child_value("headset", headset_type)

    log.info("LSL outlet ready  ·  %d channels @ %.1f Hz  ·  %s",
             len(channel_labels), srate, channel_labels)
    return StreamOutlet(info), eeg_indices


async def run_bridge() -> None:
    if CLIENT_ID.startswith("YOUR_") or CLIENT_SECRET.startswith("YOUR_"):
        log.error("Set CLIENT_ID and CLIENT_SECRET at the top of this file.")
        sys.exit(1)

    client = CortexClient(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        license=LICENSE,
        debit=DEBIT,
    )

    try:
        await client.connect()
        await client.authenticate()
        await client.find_headset()
        await client.create_session()
        cols = await client.subscribe_eeg()
    except CortexError as e:
        log.error("Setup failed: %s", e)
        await client.close()
        return
    except OSError as e:
        log.error("Could not reach Cortex at %s — is the Emotiv "
                  "Launcher running?  (%s)", CORTEX_URL, e)
        return

    outlet, eeg_idx = build_lsl_outlet(client.headset_type or "UNKNOWN", cols)

    log.info("Streaming EEG → LSL. Press Ctrl-C to stop.")
    sample_count = 0
    try:
        async for msg in client.stream():
            # Cortex EEG frames look like:
            #   {"eeg": [counter, interp, ch1, ch2, ..., raw_cq, m_hw, m],
            #    "sid": "<subscription id>",
            #    "time": 1707000000.123}
            if "eeg" in msg:
                raw = msg["eeg"]
                sample = [float(raw[i]) for i in eeg_idx]
                # `time` from Cortex is a Unix epoch; pass it through so
                # downstream tools can align with other LSL streams.
                outlet.push_sample(sample, timestamp=msg.get("time", 0.0))

                sample_count += 1
                if sample_count % 256 == 0:
                    log.info("…%d samples pushed", sample_count)

            elif "warning" in msg:
                log.warning("Cortex warning: %s", msg["warning"])
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Stop requested by user.")
    except websockets.ConnectionClosed:
        log.warning("Cortex closed the connection.")
    finally:
        await client.close()


def main() -> None:
    try:
        asyncio.run(run_bridge())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
