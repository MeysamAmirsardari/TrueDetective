"""
TrueDetective — Concealed Information Test (CIT) with P300 ERP
================================================================
A PsychoPy + LSL implementation of a three-stimulus oddball CIT, framed
around an engaging (but professional, unbiased) card-draw mechanic.

Stimulus roles
--------------
    TARGET      One constant name for the whole session, chosen by drawing
                a card in Phase 1 (the card->name mapping is hidden/random,
                so it is effectively auto-assigned). Respond UP (acknowledge).
    PROBE       The "secret" for a block. Each block begins by drawing a
                card that reveals a fresh name to remember. Respond DOWN
                (deny) — the SAME response as irrelevants — so that only the
                EEG (P300) reveals recognition.
    IRRELEVANT  Eight neutral names per block, drawn fresh from the pool.
                Respond DOWN (deny).

Each block = 10 distinct names (1 probe + 1 target + 8 irrelevants),
presented one at a time in a shuffled oddball order.

LSL markers (pushed on the exact stimulus flip)
-----------------------------------------------
    1 = probe (secret)      2 = irrelevant      3 = target
    10 = session start  11 = block start  12 = probe cue  99 = session end

Output
------
    data/behavioral_data_<timestamp>.csv  — trial-by-trial RTs, responses,
                                            accuracy, and onset timestamps.

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
    # A large master pool; each block draws its probe + irrelevants from it.
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
    "names_per_block": 10,               # 1 probe + 1 target + 8 irrelevant
    "n_blocks": 20,                      # scale up (>=30) for real EEG runs

    # ── Card-draw mechanic ─────────────────────────────────────────────
    "target_draw_n_cards": 5,            # cards shown when drawing the target
    "probe_draw_n_cards": 3,             # cards shown each block for the probe
    "use_card_draw_for_probe": True,     # False -> simple text cue instead
    "probe_view_min": 1.2,               # min seconds to view a revealed probe

    # ── Timing (seconds) ───────────────────────────────────────────────
    "fixation_min": 0.800,
    "fixation_max": 1.200,
    "stimulus_duration": 1.000,
    "response_window": 2.000,            # max wait for a key after onset
    "iti": 0.500,
    "flip_anim_frames": 38,              # card-flip animation length

    # ── Response keys ──────────────────────────────────────────────────
    "target_key": "up",                  # acknowledge the constant target
    "deny_key": "down",                  # deny probe + irrelevants
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
MARKER_PROBE = 1          # secret (deny)
MARKER_IRRELEVANT = 2     # neutral (deny)
MARKER_TARGET = 3         # constant target (acknowledge)
MARKER_SESSION_START = 10
MARKER_BLOCK_START = 11
MARKER_PROBE_CUE = 12
MARKER_SESSION_END = 99

TRIAL_TYPE_MARKER = {
    "probe": MARKER_PROBE,
    "irrelevant": MARKER_IRRELEVANT,
    "target": MARKER_TARGET,
}


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
        (MARKER_PROBE, "probe"),
        (MARKER_IRRELEVANT, "irrelevant"),
        (MARKER_TARGET, "target"),
        (MARKER_SESSION_START, "session_start"),
        (MARKER_BLOCK_START, "block_start"),
        (MARKER_PROBE_CUE, "probe_cue"),
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
# Card-draw mechanic (reusable for target and per-block probe)
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
            "First you will draw a card to receive your TARGET name for\n"
            "the whole session.\n"
            "    •  When your target name appears, press the UP arrow.\n"
            "    •  For every other name, press the DOWN arrow.\n\n"
            "Each block then begins by drawing a card that reveals one\n"
            "name to remember. During that block, respond to it with the\n"
            "DOWN arrow, the same as the other non-target names.\n\n"
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


def show_target_rule(win, target_name):
    """Reinforce the response rule after the target has been drawn."""
    caption = visual.TextStim(
        win, text="YOUR TARGET NAME FOR THIS SESSION", pos=(0, 0.18),
        height=0.026, color=CONFIG["color_dim"], font=CONFIG["font_body"],
    )
    name = visual.TextStim(
        win, text=target_name, pos=(0, 0.04), height=0.12,
        color=CONFIG["color_accent"], font=CONFIG["font_display"], bold=True,
    )
    rule = visual.TextStim(
        win,
        text=(
            f"Press the UP arrow every time {target_name} appears.\n"
            "Press the DOWN arrow for all other names."
        ),
        pos=(0, -0.16), height=0.028,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        alignText="center", wrapWidth=1.2,
    )
    while True:
        header(win, "TARGET CONFIRMED")
        panel(win, width=1.25, height=0.7)
        caption.draw()
        name.draw()
        rule.draw()
        footer(win, "Press SPACE to begin   ·   ESC to quit")
        win.flip()
        wait_for_keys([CONFIG["advance_key"]])
        return


def show_probe_cue_text(win, probe_name, block_idx, n_blocks, outlet):
    """Fallback non-card probe cue (used when card draw is disabled)."""
    outlet.push_sample([MARKER_PROBE_CUE])
    caption = visual.TextStim(
        win, text="REMEMBER THIS NAME", pos=(0, 0.16), height=0.026,
        color=CONFIG["color_dim"], font=CONFIG["font_body"],
    )
    name = visual.TextStim(
        win, text=probe_name, pos=(0, 0.0), height=0.11,
        color=CONFIG["color_text"], font=CONFIG["font_display"], bold=True,
    )
    while True:
        header(win, f"BLOCK {block_idx} / {n_blocks}")
        panel(win, width=1.1, height=0.5)
        caption.draw()
        name.draw()
        footer(win, "Press SPACE when ready   ·   ESC to quit")
        win.flip()
        wait_for_keys([CONFIG["advance_key"]], min_wait=CONFIG["probe_view_min"])
        return


# =============================================================================
# Block construction
# =============================================================================

def build_block(target_name, probe_name, pool):
    """
    Build one block's sequence given the (already drawn) probe and the
    constant target: probe + target + 8 fresh irrelevants, all distinct,
    shuffled. Returns the (name, trial_type) sequence.
    """
    n_irrelevant = CONFIG["names_per_block"] - 2
    available = [n for n in pool if n not in (target_name, probe_name)]
    irrelevants = random.sample(available, n_irrelevant)

    sequence = [(probe_name, "probe"), (target_name, "target")]
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


def expected_key(trial_type):
    return CONFIG["target_key"] if trial_type == "target" else CONFIG["deny_key"]


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
            keyList=[CONFIG["target_key"], CONFIG["deny_key"],
                     CONFIG["quit_key"]],
            timeStamped=rt_clock,
        )
        if keys:
            key_name, key_time = keys[0]
            if key_name == CONFIG["quit_key"]:
                raise KeyboardInterrupt("User pressed escape.")
            response, rt = key_name, key_time
            break

    if stim_visible:
        win.flip()

    # ── ITI ────────────────────────────────────────────────────────────
    iti_clock = core.Clock()
    while iti_clock.getTime() < CONFIG["iti"]:
        win.flip()
        check_quit()

    exp_key = expected_key(trial_type)
    correct = (response == exp_key)
    return {
        "stimulus": name,
        "trial_type": trial_type,
        "marker_code": marker_code,
        "expected_key": exp_key,
        "response": response if response is not None else "none",
        "correct": int(correct),
        "rt_seconds": round(rt, 4) if rt is not None else "",
        "stim_onset_psychopy": round(flip_time, 5),
        "stim_onset_lsl": round(lsl_stamp, 5),
    }


def draw_block_probe(win, mouse, target_name, block_idx, n_blocks, outlet):
    """Draw the block's secret via the card mechanic (or text fallback)."""
    pool = CONFIG["master_name_pool"]
    candidates = [n for n in pool if n != target_name]

    if CONFIG["use_card_draw_for_probe"]:
        return run_card_draw(
            win, mouse,
            candidate_pool=candidates,
            n_cards=CONFIG["probe_draw_n_cards"],
            header_label=f"BLOCK {block_idx} / {n_blocks}  ·  DRAW THE NAME",
            prompt_text="Draw a card to reveal the name to remember.",
            reveal_caption="Remember this name for the upcoming block.",
            outlet=outlet, reveal_marker=MARKER_PROBE_CUE,
            min_view=CONFIG["probe_view_min"],
        )
    # Fallback: pick randomly and show a plain text cue.
    probe_name = random.choice(candidates)
    show_probe_cue_text(win, probe_name, block_idx, n_blocks, outlet)
    return probe_name


def run_session(win, mouse, outlet, target_name):
    outlet.push_sample([MARKER_SESSION_START])
    pool = CONFIG["master_name_pool"]
    n_blocks = CONFIG["n_blocks"]

    stim_text = visual.TextStim(
        win, text="", pos=(0, 0), height=0.12,
        color=CONFIG["color_text"], font=CONFIG["font_display"], bold=True,
    )

    results = []
    for block_idx in range(1, n_blocks + 1):
        probe_name = draw_block_probe(
            win, mouse, target_name, block_idx, n_blocks, outlet
        )
        sequence = build_block(target_name, probe_name, pool)

        outlet.push_sample([MARKER_BLOCK_START])
        for pos, (name, trial_type) in enumerate(sequence, start=1):
            check_quit()
            row = run_trial(win, outlet, stim_text, name, trial_type)
            row.update({
                "block": block_idx,
                "position_in_block": pos,
                "block_probe": probe_name,
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
        "marker_code", "block_probe", "session_target",
        "expected_key", "response", "correct", "rt_seconds",
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
        # Phase 1 — draw the constant target from a deck of cards.
        target_name = run_card_draw(
            win, mouse,
            candidate_pool=CONFIG["master_name_pool"],
            n_cards=CONFIG["target_draw_n_cards"],
            header_label="DRAW YOUR TARGET",
            prompt_text="Draw a card to receive your target name.",
            reveal_caption="This is your target name for the whole session.",
            min_view=CONFIG["probe_view_min"],
        )
        show_target_rule(win, target_name)

        results = run_session(win, mouse, outlet, target_name)
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
