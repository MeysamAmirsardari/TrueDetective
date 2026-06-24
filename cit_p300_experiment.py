"""
TrueDetective — Concealed Information Test (CIT) with P300 ERP
================================================================
A PsychoPy + LSL implementation of a three-stimulus oddball CIT, framed
around an engaging (but professional, unbiased) card-draw mechanic.

Stimulus roles
--------------
    TARGET      One constant name for the whole session, chosen by drawing
                a card in Phase 1 (the card->name mapping is hidden/random,
                so it is effectively auto-assigned). Acknowledge it with the
                TARGET key (RIGHT or UP arrow — either works).
    SECRET      Your constant "secret", drawn once at the start (a second
                card draw). Deny it with the DENY key (LEFT or DOWN arrow) —
                the SAME response as irrelevants — so that only the EEG (P300)
                reveals recognition.
    IRRELEVANT  A fixed set of eight neutral names, drawn once at the start
                and reused every block so all ten names appear equally often.
                Deny (LEFT or DOWN arrow).

Each block = the same 10 names (1 secret + 1 target + 8 irrelevants),
re-shuffled into a fresh oddball order. Every name is shown exactly once per
block, so the secret, target, and each irrelevant are frequency-balanced.

LSL markers (pushed on the exact stimulus flip)
-----------------------------------------------
    1 = secret      2 = irrelevant      3 = target
    10 = session start  11 = block start  12 = secret cue  99 = session end

Output
------
    data/behavioral_data_<timestamp>.csv  — trial-by-trial RTs, responses,
                                            accuracy, and onset timestamps.
                                            Responses are key-agnostic codes:
                                            1 = target acknowledged, 0 = denied
                                            (so LEFT/RIGHT and UP/DOWN are
                                            interchangeable).

Install once:
    pip install psychopy pylsl numpy
"""

import csv
import os
import random
from datetime import datetime

import numpy as np
from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet, local_clock

# =============================================================================
# CONFIGURATION  — tweak freely
# =============================================================================

CONFIG = {
    # ── Window ──────────────────────────────────────────────────────────
    "fullscreen": True,
    "screen_size": (1440, 900),          # used only if fullscreen=False
    "monitor_name": "testMonitor",

    # ── Names ───────────────────────────────────────────────────────────
    # A large master pool; the secret, target, and the fixed irrelevant set
    # are all drawn from it once at the start of the session.
    "master_name_pool": [
        "LIAM", "OLIVIA", "NOAH", "EMMA", "OLIVER", "AVA", "ELIJAH", "SOPHIA",
        "JAMES", "ISABELLA", "WILLIAM", "MIA", "BENJAMIN", "CHARLOTTE",
        "LUCAS", "AMELIA", "HENRY", "HARPER", "ALEXANDER", "EVELYN", "MASON",
        "ABIGAIL", "ETHAN", "EMILY", "DANIEL", "ELLA", "JACOB", "ELIZABETH",
        "LOGAN", "CAMILA", "JACKSON", "LUNA", "LEVI", "SOFIA", "SEBASTIAN",
        "AVERY", "MATEO", "MILA", "JACK", "ARIA", "OWEN", "SCARLETT",
        "THEODORE", "PENELOPE", "AIDEN", "LAYLA", "SAMUEL", "NORA", "JOSEPH",
        "ZOEY",
    ],

    # ── Block / trial structure ────────────────────────────────────────
    "names_per_block": 10,               # 1 secret + 1 target + 8 irrelevant
    "n_blocks": 20,                      # scale up (>=30) for real EEG runs

    # ── Card-draw mechanic (target + secret drawn once, at the start) ──
    "target_draw_n_cards": 5,            # cards shown when drawing the target
    "secret_draw_n_cards": 5,            # cards shown when drawing the secret
    "reveal_view_min": 1.2,              # min seconds to view a revealed name

    # ── Timing (seconds) ───────────────────────────────────────────────
    "fixation_min": 0.800,
    "fixation_max": 1.200,
    "stimulus_duration": 1.000,
    "response_window": 2.000,            # max wait for a key after onset
    "iti": 0.500,
    "flip_anim_frames": 38,              # card-flip animation length

    # ── Response keys (either arrow pair works) ────────────────────────
    # Target = RIGHT or UP; deny = LEFT or DOWN. Responses are logged by
    # meaning (1/0), not by physical key, so the two pairs are interchangeable.
    "target_keys": ["right", "up"],      # acknowledge the constant target
    "deny_keys": ["left", "down"],       # deny secret + irrelevants
    "advance_key": "space",
    "quit_key": "escape",

    # ── Professional, neutral palette (no biasing theme) ───────────────
    "color_bg": "#1A1D23",               # dark slate
    "color_panel": "#23272F",
    "color_card_back": "#2A2F38",
    "color_card_hover": "#323845",
    "color_text": "#E8EAED",
    "color_dim": "#9AA0A6",
    "color_accent": "#4C8BF5",           # restrained blue
    "color_rule": "#3A3F48",
    "color_fixation": "#C8CDD4",         # neutral light-gray cross

    # ── Fonts ──────────────────────────────────────────────────────────
    "font_display": "Arial",
    "font_body": "Arial",

    # ── LSL ────────────────────────────────────────────────────────────
    "lsl_stream_name": "TrueDetective_Markers",
    "lsl_stream_type": "Markers",
    "lsl_source_id": "truedetective_cit_v1",
}

# Marker codes
MARKER_SECRET = 1         # secret (deny)
MARKER_IRRELEVANT = 2     # neutral (deny)
MARKER_TARGET = 3         # constant target (acknowledge)
MARKER_SESSION_START = 10
MARKER_BLOCK_START = 11
MARKER_SECRET_CUE = 12
MARKER_SESSION_END = 99

TRIAL_TYPE_MARKER = {
    "secret": MARKER_SECRET,
    "irrelevant": MARKER_IRRELEVANT,
    "target": MARKER_TARGET,
}

# Key-agnostic response codes written to the CSV. Recording the *meaning* of
# the press (rather than "left"/"right"/"up"/"down") keeps the data identical
# no matter which arrow pair the participant used.
RESPONSE_TARGET = 1       # acknowledged as the target
RESPONSE_DENY = 0         # denied (secret or irrelevant)


# =============================================================================
# LSL — marker outlet
# =============================================================================

def init_lsl_outlet():
    """Create an int32 marker StreamOutlet for EEG synchronisation."""
    info = StreamInfo(
        name=CONFIG["lsl_stream_name"],
        type=CONFIG["lsl_stream_type"],
        channel_count=1,
        nominal_srate=0,                 # irregular
        channel_format="int32",
        source_id=CONFIG["lsl_source_id"],
    )
    markers = info.desc().append_child("markers")
    for code, label in [
        (MARKER_SECRET, "secret"),
        (MARKER_IRRELEVANT, "irrelevant"),
        (MARKER_TARGET, "target"),
        (MARKER_SESSION_START, "session_start"),
        (MARKER_BLOCK_START, "block_start"),
        (MARKER_SECRET_CUE, "secret_cue"),
        (MARKER_SESSION_END, "session_end"),
    ]:
        m = markers.append_child("marker")
        m.append_child_value("code", str(code))
        m.append_child_value("label", label)
    return StreamOutlet(info)


# =============================================================================
# Window & reusable visual helpers
# =============================================================================

def make_window():
    return visual.Window(
        size=CONFIG["screen_size"],
        fullscr=CONFIG["fullscreen"],
        monitor=CONFIG["monitor_name"],
        color=CONFIG["color_bg"],
        colorSpace="rgb",
        units="height",
        allowGUI=False,
        winType="pyglet",
    )


def header(win, label):
    """A thin, neutral header line with a small caption."""
    aspect = win.size[0] / win.size[1]
    visual.Line(
        win, start=(-aspect / 2 + 0.05, 0.44), end=(aspect / 2 - 0.05, 0.44),
        lineColor=CONFIG["color_rule"], lineWidth=1.0,
    ).draw()
    visual.TextStim(
        win, text=label, pos=(-aspect / 2 + 0.06, 0.465), height=0.020,
        color=CONFIG["color_dim"], font=CONFIG["font_body"],
        anchorHoriz="left", alignText="left",
    ).draw()


def footer(win, text):
    visual.TextStim(
        win, text=text, pos=(0, -0.46), height=0.018,
        color=CONFIG["color_dim"], font=CONFIG["font_body"],
    ).draw()


def panel(win, width=1.25, height=0.7, line_color=None):
    visual.Rect(
        win, width=width, height=height, pos=(0, 0),
        fillColor=CONFIG["color_panel"],
        lineColor=line_color or CONFIG["color_rule"],
        lineWidth=1.0, opacity=1.0,
    ).draw()


def check_quit():
    if event.getKeys(keyList=[CONFIG["quit_key"]]):
        raise KeyboardInterrupt("User pressed escape.")


def wait_for_keys(keys, min_wait=0.0):
    """Block until one of `keys` is pressed; abort on escape."""
    clock = core.Clock()
    event.clearEvents()
    while True:
        pressed = event.getKeys(keyList=list(keys) + [CONFIG["quit_key"]])
        if pressed:
            if CONFIG["quit_key"] in pressed:
                raise KeyboardInterrupt("User pressed escape.")
            if clock.getTime() >= min_wait:
                return pressed[0]
            event.clearEvents()         # too early — keep waiting
        core.wait(0.005)


# =============================================================================
# Card-draw mechanic (reusable for both the target and the secret draws)
# =============================================================================

def _build_cards(win, names):
    """Lay out N face-down cards evenly, return a list of card dicts."""
    n = len(names)
    card_w, card_h, gap = 0.16, 0.24, 0.045
    total_w = n * card_w + (n - 1) * gap
    start_x = -total_w / 2 + card_w / 2
    cards = []
    for i, name in enumerate(names):
        pos = (start_x + i * (card_w + gap), -0.02)
        cards.append({
            "name": name,
            "pos": pos,
            "w": card_w,
            "h": card_h,
            "rect": visual.Rect(
                win, width=card_w, height=card_h, pos=pos,
                fillColor=CONFIG["color_card_back"],
                lineColor=CONFIG["color_rule"], lineWidth=2.0,
            ),
            # subtle diamond emblem on the card back (neutral motif)
            "emblem": visual.Rect(
                win, width=0.05, height=0.05, pos=pos, ori=45,
                fillColor=None, lineColor=CONFIG["color_dim"], lineWidth=1.5,
                opacity=0.6,
            ),
            "label": visual.TextStim(
                win, text=name, pos=pos, height=0.040,
                color=CONFIG["color_accent"], font=CONFIG["font_display"],
                bold=True,
            ),
        })
    return cards


def _draw_facedown(cards, mouse, hover_enabled=True):
    """Render all cards face-down, with hover highlight on the one under
    the cursor. Returns the hovered index (or None)."""
    hovered = None
    for i, c in enumerate(cards):
        is_hover = hover_enabled and c["rect"].contains(mouse)
        if is_hover:
            hovered = i
            c["rect"].fillColor = CONFIG["color_card_hover"]
            c["rect"].lineColor = CONFIG["color_accent"]
            c["rect"].lineWidth = 3.0
        else:
            c["rect"].fillColor = CONFIG["color_card_back"]
            c["rect"].lineColor = CONFIG["color_rule"]
            c["rect"].lineWidth = 2.0
        c["rect"].draw()
        c["emblem"].draw()
    return hovered


def run_card_draw(win, mouse, candidate_pool, n_cards, header_label,
                  prompt_text, reveal_caption, outlet=None,
                  reveal_marker=None, min_view=0.0):
    """
    Present `n_cards` face-down cards drawn from `candidate_pool`. The user
    clicks one; it flips open to reveal the (randomly assigned, hidden) name
    behind it. Returns the revealed name.

    The card->name mapping is random and concealed, so the participant's
    free choice still yields a random assignment.
    """
    names = random.sample(candidate_pool, n_cards)
    cards = _build_cards(win, names)

    prompt = visual.TextStim(
        win, text=prompt_text, pos=(0, 0.30), height=0.034,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        alignText="center", wrapWidth=1.3,
    )

    mouse.setVisible(True)
    mouse.clickReset()

    # ── Selection loop ─────────────────────────────────────────────────
    chosen = None
    while chosen is None:
        check_quit()
        header(win, header_label)
        prompt.draw()
        _draw_facedown(cards, mouse)
        footer(win, "Click a card to select   ·   ESC to quit")
        win.flip()

        if mouse.getPressed()[0]:
            for i, c in enumerate(cards):
                if c["rect"].contains(mouse):
                    chosen = i
                    break
            while mouse.getPressed()[0]:    # debounce release
                core.wait(0.005)

    mouse.setVisible(False)
    chosen_name = cards[chosen]["name"]

    # ── Flip / reveal animation ────────────────────────────────────────
    n_frames = CONFIG["flip_anim_frames"]
    for f in range(n_frames + 1):
        progress = f / n_frames
        scale = max(abs(np.cos(progress * np.pi)), 0.02)
        flipped = progress > 0.5

        header(win, header_label)
        prompt.draw()
        for i, c in enumerate(cards):
            if i == chosen:
                c["rect"].width = c["w"] * scale
                if flipped:
                    c["rect"].fillColor = CONFIG["color_panel"]
                    c["rect"].lineColor = CONFIG["color_accent"]
                    c["rect"].lineWidth = 3.0
                    c["rect"].draw()
                    c["label"].opacity = (progress - 0.5) * 2
                    c["label"].draw()
                else:
                    c["rect"].fillColor = CONFIG["color_card_hover"]
                    c["rect"].draw()
                    c["emblem"].opacity = max(0.0, 0.6 - progress * 1.2)
                    c["emblem"].draw()
            else:
                c["rect"].opacity = max(0.25, 1 - progress)
                c["rect"].draw()
                c["emblem"].opacity = max(0.0, 0.6 - progress)
                c["emblem"].draw()
                c["rect"].opacity = 1.0
        win.flip()

        # Push the cue marker right as the name becomes visible.
        if outlet is not None and reveal_marker is not None and f == n_frames // 2:
            outlet.push_sample([reveal_marker])

    cards[chosen]["rect"].width = cards[chosen]["w"]

    # ── Hold the reveal so the participant can read / memorise it ──────
    caption = visual.TextStim(
        win, text=reveal_caption, pos=(0, 0.30), height=0.030,
        color=CONFIG["color_dim"], font=CONFIG["font_body"],
        alignText="center", wrapWidth=1.3,
    )
    big_name = visual.TextStim(
        win, text=chosen_name, pos=(0, -0.02), height=0.075,
        color=CONFIG["color_accent"], font=CONFIG["font_display"], bold=True,
    )
    while True:
        header(win, header_label)
        caption.draw()
        big_name.draw()
        footer(win, "Press SPACE to continue   ·   ESC to quit")
        win.flip()
        wait_for_keys([CONFIG["advance_key"]], min_wait=min_view)
        return chosen_name


# =============================================================================
# Instruction & assignment screens
# =============================================================================

def show_instructions(win):
    title = visual.TextStim(
        win, text="Name Recognition Task", pos=(0, 0.32), height=0.052,
        color=CONFIG["color_text"], font=CONFIG["font_display"], bold=True,
    )
    body = visual.TextStim(
        win,
        text=(
            "First you will draw two cards, one after the other:\n"
            "    •  a TARGET name : press the RIGHT arrow whenever it appears.\n"
            "    •  a name to REMEMBER : press the LEFT arrow for it, the\n"
            "       same as every other name.\n\n"
            "Both names stay the same for the whole session.\n\n"
            "After that, names appear one at a time. Press RIGHT only for your\n"
            "target name; press LEFT for all other names.\n\n"
            "Please respond as quickly and accurately as you can.\n"
            "Keep your gaze on the central cross and try to stay still."
        ),
        pos=(0, -0.04), height=0.028,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        wrapWidth=1.3, alignText="left",
    )
    while True:
        header(win, "INSTRUCTIONS")
        panel(win, width=1.4, height=0.78)
        title.draw()
        body.draw()
        footer(win, "Press SPACE to continue   ·   ESC to quit")
        win.flip()
        wait_for_keys([CONFIG["advance_key"]])
        return


def show_rules(win, target_name, secret_name):
    """Confirm the response rule for both the target and the secret."""
    target_label = visual.TextStim(
        win, text="TARGET NAME  :  press the RIGHT arrow", pos=(0, 0.24),
        height=0.024, color=CONFIG["color_dim"], font=CONFIG["font_body"],
    )
    target_stim = visual.TextStim(
        win, text=target_name, pos=(0, 0.15), height=0.085,
        color=CONFIG["color_accent"], font=CONFIG["font_display"], bold=True,
    )
    secret_label = visual.TextStim(
        win, text="NAME TO REMEMBER  :  press the LEFT arrow", pos=(0, -0.04),
        height=0.024, color=CONFIG["color_dim"], font=CONFIG["font_body"],
    )
    secret_stim = visual.TextStim(
        win, text=secret_name, pos=(0, -0.13), height=0.085,
        color=CONFIG["color_text"], font=CONFIG["font_display"], bold=True,
    )
    note = visual.TextStim(
        win,
        text="Press LEFT for every other name as well.",
        pos=(0, -0.26), height=0.024,
        color=CONFIG["color_dim"], font=CONFIG["font_body"],
    )
    while True:
        header(win, "RESPONSE RULES")
        panel(win, width=1.3, height=0.78)
        target_label.draw()
        target_stim.draw()
        secret_label.draw()
        secret_stim.draw()
        note.draw()
        footer(win, "Press SPACE to begin   ·   ESC to quit")
        win.flip()
        wait_for_keys([CONFIG["advance_key"]])
        return


# =============================================================================
# Block construction
# =============================================================================

def build_block(target_name, secret_name, irrelevants):
    """
    Build one block's sequence from the constant secret, the constant target,
    and the fixed list of irrelevants. The same ten names are reused in every
    block — only their order is reshuffled — so every name is shown exactly
    once per block and is therefore frequency-balanced across the session.
    Returns the (name, trial_type) sequence.
    """
    sequence = [(secret_name, "secret"), (target_name, "target")]
    sequence += [(name, "irrelevant") for name in irrelevants]
    random.shuffle(sequence)
    return sequence


# =============================================================================
# Trial / block runtime
# =============================================================================

def stylized_fixation(win):
    visual.TextStim(
        win, text="+", pos=(0, 0), height=0.06,
        color=CONFIG["color_fixation"], font=CONFIG["font_display"], bold=True,
    ).draw()


def response_keys():
    """Every accepted response key (both arrow pairs)."""
    return CONFIG["target_keys"] + CONFIG["deny_keys"]


def response_code(key):
    """Map a physical key to its key-agnostic response code (1/0), or None."""
    if key in CONFIG["target_keys"]:
        return RESPONSE_TARGET
    if key in CONFIG["deny_keys"]:
        return RESPONSE_DENY
    return None


def expected_response(trial_type):
    """Correct response code for a trial type: target → 1, otherwise → 0."""
    return RESPONSE_TARGET if trial_type == "target" else RESPONSE_DENY


def run_trial(win, outlet, stim_text, name, trial_type):
    """
    One stimulus trial: fixation → stimulus (LSL marker on flip) →
    response collection → ITI. Returns a dict of behavioural data.
    """
    rt_clock = core.Clock()

    # ── Fixation (jittered) ────────────────────────────────────────────
    fix_dur = random.uniform(CONFIG["fixation_min"], CONFIG["fixation_max"])
    fix_clock = core.Clock()
    while fix_clock.getTime() < fix_dur:
        stylized_fixation(win)
        win.flip()
        check_quit()

    # ── Stimulus — push the LSL marker on the exact flip ───────────────
    stim_text.text = name
    marker_code = TRIAL_TYPE_MARKER[trial_type]
    event.clearEvents()
    stim_text.draw()
    win.callOnFlip(outlet.push_sample, [marker_code])
    win.callOnFlip(rt_clock.reset)
    flip_time = win.flip()
    lsl_stamp = local_clock()

    # ── Response collection ────────────────────────────────────────────
    response, rt = None, None
    stim_offset = flip_time + CONFIG["stimulus_duration"]
    deadline = flip_time + CONFIG["response_window"]
    stim_visible = True

    while core.getTime() < deadline:
        if stim_visible and core.getTime() >= stim_offset:
            win.flip()                    # clear stimulus, keep listening
            stim_visible = False
        elif stim_visible:
            stim_text.draw()
            win.flip()

        keys = event.getKeys(
            keyList=response_keys() + [CONFIG["quit_key"]],
            timeStamped=rt_clock,
        )
        if keys:
            key_name, key_time = keys[0]
            if key_name == CONFIG["quit_key"]:
                raise KeyboardInterrupt("User pressed escape.")
            response, rt = response_code(key_name), key_time
            break

    if stim_visible:
        win.flip()

    # ── ITI ────────────────────────────────────────────────────────────
    iti_clock = core.Clock()
    while iti_clock.getTime() < CONFIG["iti"]:
        win.flip()
        check_quit()

    exp_code = expected_response(trial_type)
    correct = (response == exp_code)
    return {
        "stimulus": name,
        "trial_type": trial_type,
        "marker_code": marker_code,
        "expected_response": exp_code,
        "response": response if response is not None else "none",
        "correct": int(correct),
        "rt_seconds": round(rt, 4) if rt is not None else "",
        "stim_onset_psychopy": round(flip_time, 5),
        "stim_onset_lsl": round(lsl_stamp, 5),
    }


def run_session(win, outlet, target_name, secret_name):
    """Run all blocks. The secret, the target, and the eight irrelevants are
    all fixed for the whole session; only their order is reshuffled each block,
    so every name appears exactly once per block (frequency-balanced)."""
    outlet.push_sample([MARKER_SESSION_START])
    pool = CONFIG["master_name_pool"]
    n_blocks = CONFIG["n_blocks"]

    # Fixed irrelevant set, drawn once so every name is equally frequent.
    n_irrelevant = CONFIG["names_per_block"] - 2
    available = [n for n in pool if n not in (target_name, secret_name)]
    irrelevants = random.sample(available, n_irrelevant)

    stim_text = visual.TextStim(
        win, text="", pos=(0, 0), height=0.12,
        color=CONFIG["color_text"], font=CONFIG["font_display"], bold=True,
    )

    results = []
    for block_idx in range(1, n_blocks + 1):
        sequence = build_block(target_name, secret_name, irrelevants)

        outlet.push_sample([MARKER_BLOCK_START])
        for pos, (name, trial_type) in enumerate(sequence, start=1):
            check_quit()
            row = run_trial(win, outlet, stim_text, name, trial_type)
            row.update({
                "block": block_idx,
                "position_in_block": pos,
                "session_secret": secret_name,
                "session_target": target_name,
            })
            results.append(row)

    outlet.push_sample([MARKER_SESSION_END])
    return results


# =============================================================================
# Data persistence
# =============================================================================

def save_results(results, target_name):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(out_dir, f"behavioral_data_{timestamp}.csv")

    fieldnames = [
        "block", "position_in_block", "stimulus", "trial_type",
        "marker_code", "session_secret", "session_target",
        "expected_response", "response", "correct", "rt_seconds",
        "stim_onset_psychopy", "stim_onset_lsl",
    ]
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return fname


def show_goodbye(win, csv_path, n_trials):
    title = visual.TextStim(
        win, text="Session complete", pos=(0, 0.14), height=0.05,
        color=CONFIG["color_text"], font=CONFIG["font_display"], bold=True,
    )
    body = visual.TextStim(
        win,
        text=(
            f"{n_trials} trials recorded.\n"
            f"Data saved to: {os.path.basename(csv_path)}\n\n"
            "Thank you for participating."
        ),
        pos=(0, -0.06), height=0.028,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        alignText="center", wrapWidth=1.1,
    )
    header(win, "COMPLETE")
    panel(win, width=1.2, height=0.5)
    title.draw()
    body.draw()
    footer(win, "Press any key to exit")
    win.flip()
    event.waitKeys()


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed()
    outlet = init_lsl_outlet()
    core.wait(0.5)                        # let LSL consumers subscribe

    win = make_window()
    mouse = event.Mouse(visible=False, win=win)
    results, target_name, csv_path = [], None, None

    try:
        show_instructions(win)
        # Phase 1 — draw the constant target, then the constant secret.
        target_name = run_card_draw(
            win, mouse,
            candidate_pool=CONFIG["master_name_pool"],
            n_cards=CONFIG["target_draw_n_cards"],
            header_label="DRAW YOUR TARGET",
            prompt_text="Draw a card to receive your target name.",
            reveal_caption="This is your target name for the whole session.",
            min_view=CONFIG["reveal_view_min"],
        )
        secret_name = run_card_draw(
            win, mouse,
            candidate_pool=[n for n in CONFIG["master_name_pool"]
                            if n != target_name],
            n_cards=CONFIG["secret_draw_n_cards"],
            header_label="DRAW A NAME TO REMEMBER",
            prompt_text="Draw a card to receive the name to remember.",
            reveal_caption=("Remember this name. Respond to it with the "
                            "LEFT arrow, the same as the other names."),
            outlet=outlet, reveal_marker=MARKER_SECRET_CUE,
            min_view=CONFIG["reveal_view_min"],
        )
        show_rules(win, target_name, secret_name)

        results = run_session(win, outlet, target_name, secret_name)
        csv_path = save_results(results, target_name)
        show_goodbye(win, csv_path, len(results))

    except KeyboardInterrupt:
        if results:
            csv_path = save_results(results, target_name or "ABORTED")
            print(f"[TrueDetective] Aborted. Partial data saved to {csv_path}")
        else:
            print("[TrueDetective] Aborted before any trials were run.")

    finally:
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
