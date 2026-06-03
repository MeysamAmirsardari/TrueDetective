"""
TrueDetective — Concealed Information Test (CIT) with P300 ERP
================================================================
A PsychoPy + LSL implementation of a classic oddball CIT paradigm for
deception detection using an Emotiv EEG headset.

Pipeline:
    Phase 1 — Digital Lottery: participant clicks one of five face-down
              cards; the revealed name becomes the Probe (Target).
    Phase 2 — Oddball CIT: 100 trials, 20% Probe / 80% Irrelevant.
              LSL markers are pushed on the exact screen flip when the
              stimulus appears (marker 1 = Probe/Lie, 2 = Irrelevant/Truth).

Output:
    behavioral_data_<timestamp>.csv  — trial-by-trial RTs and responses.

Install once:
    pip install psychopy pylsl pandas numpy
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
    # Window
    "fullscreen": True,
    "screen_size": (1440, 900),          # used only if fullscreen=False
    "monitor_name": "testMonitor",

    # Stimuli
    "name_pool": ["SARA", "JOHN", "MARK", "LISA", "EMMA"],
    "n_trials": 100,
    "target_ratio": 0.20,                # 20% probes (oddball)

    # Timing (seconds)
    "fixation_min": 0.800,
    "fixation_max": 1.200,
    "stimulus_duration": 1.000,
    "response_window": 2.000,            # max wait for the Down key
    "iti": 0.500,

    # Response
    "response_key": "down",
    "quit_key": "escape",

    # Cyber-interrogation palette
    "color_bg": "#0A0E14",               # near-black
    "color_panel": "#11161F",
    "color_text": "#E6EDF3",
    "color_dim": "#7D8590",
    "color_accent": "#00E5FF",           # neon cyan
    "color_accent_2": "#FF3864",         # neon magenta (probe reveal)
    "color_card_face": "#1B2230",
    "color_card_border": "#2A3242",
    "color_fixation": "#00E5FF",

    # Fonts (PsychoPy falls back gracefully if unavailable)
    "font_display": "Consolas",
    "font_body": "Helvetica",

    # LSL
    "lsl_stream_name": "TrueDetective_Markers",
    "lsl_stream_type": "Markers",
    "lsl_source_id": "truedetective_cit_v1",
}

MARKER_PROBE = 1        # target / lie
MARKER_IRRELEVANT = 2   # neutral / truth
MARKER_PHASE1_START = 10
MARKER_PHASE2_START = 11
MARKER_EXPERIMENT_END = 99


# =============================================================================
# LSL — Lab Streaming Layer marker outlet
# =============================================================================

def init_lsl_outlet():
    """Create an int32 marker StreamOutlet for sync with the EEG recorder."""
    info = StreamInfo(
        name=CONFIG["lsl_stream_name"],
        type=CONFIG["lsl_stream_type"],
        channel_count=1,
        nominal_srate=0,                 # irregular
        channel_format="int32",
        source_id=CONFIG["lsl_source_id"],
    )
    # Metadata so downstream analysis can decode the marker codes.
    desc = info.desc()
    desc.append_child_value("manufacturer", "TrueDetective")
    markers = desc.append_child("markers")
    for code, label in [
        (MARKER_PROBE, "probe"),
        (MARKER_IRRELEVANT, "irrelevant"),
        (MARKER_PHASE1_START, "phase1_start"),
        (MARKER_PHASE2_START, "phase2_start"),
        (MARKER_EXPERIMENT_END, "experiment_end"),
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


def draw_backdrop(win):
    """Subtle 'interrogation grid' backdrop — thin dim lines."""
    aspect = win.size[0] / win.size[1]
    for x in np.linspace(-aspect / 2, aspect / 2, 13):
        visual.Line(
            win, start=(x, -0.5), end=(x, 0.5),
            lineColor=CONFIG["color_panel"], lineWidth=1.0,
        ).draw()
    for y in np.linspace(-0.5, 0.5, 9):
        visual.Line(
            win, start=(-aspect / 2, y), end=(aspect / 2, y),
            lineColor=CONFIG["color_panel"], lineWidth=1.0,
        ).draw()


def header_bar(win, label="TRUEDETECTIVE // CIT-P300"):
    aspect = win.size[0] / win.size[1]
    visual.Rect(
        win, width=aspect, height=0.06, pos=(0, 0.47),
        fillColor=CONFIG["color_panel"], lineColor=CONFIG["color_accent"],
        lineWidth=1.0,
    ).draw()
    visual.TextStim(
        win, text=label, pos=(0, 0.47), height=0.022,
        color=CONFIG["color_accent"], font=CONFIG["font_display"],
        bold=True, anchorHoriz="center",
    ).draw()


def footer_hint(win, text):
    visual.TextStim(
        win, text=text, pos=(0, -0.46), height=0.018,
        color=CONFIG["color_dim"], font=CONFIG["font_body"],
    ).draw()


def check_quit():
    if CONFIG["quit_key"] in event.getKeys(keyList=[CONFIG["quit_key"]]):
        raise KeyboardInterrupt("User pressed escape.")


def wait_for_keys(keys):
    """Block until one of `keys` is pressed; abort on escape."""
    event.clearEvents()
    while True:
        pressed = event.getKeys(keyList=list(keys) + [CONFIG["quit_key"]])
        if pressed:
            if CONFIG["quit_key"] in pressed:
                raise KeyboardInterrupt("User pressed escape.")
            return pressed[0]
        core.wait(0.01)


# =============================================================================
# Phase 0 — Intro screen
# =============================================================================

def show_intro(win):
    title = visual.TextStim(
        win, text="T R U E   D E T E C T I V E",
        pos=(0, 0.30), height=0.075,
        color=CONFIG["color_accent"], font=CONFIG["font_display"], bold=True,
    )
    subtitle = visual.TextStim(
        win, text="Concealed Information Test  ·  P300 Oddball Paradigm",
        pos=(0, 0.21), height=0.025,
        color=CONFIG["color_dim"], font=CONFIG["font_body"],
    )

    body = (
        "Welcome, suspect.\n\n"
        "You are about to draw a hidden identity from a deck of five cards.\n"
        "That name will become your secret — the one piece of information\n"
        "you must conceal from the system that is about to read your brain.\n\n"
        "The machine will show you many names, one at a time.\n"
        "Press the DOWN ARROW for every single name, no exceptions —\n"
        "as if none of them mean anything to you.\n\n"
        "Your brain will betray you. Or it won't.\n"
        "Let's find out."
    )
    body_stim = visual.TextStim(
        win, text=body, pos=(0, -0.05), height=0.028,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        wrapWidth=1.2, alignText="center",
    )

    box = visual.Rect(
        win, width=1.35, height=0.85, pos=(0, 0),
        fillColor=CONFIG["color_panel"],
        lineColor=CONFIG["color_accent"], lineWidth=1.5, opacity=0.55,
    )

    while True:
        draw_backdrop(win)
        header_bar(win)
        box.draw()
        title.draw()
        subtitle.draw()
        body_stim.draw()
        footer_hint(win, "Press [SPACE] to begin  ·  [ESC] to abort")
        win.flip()
        wait_for_keys(["space"])
        return


# =============================================================================
# Phase 1 — Digital Lottery (card selection)
# =============================================================================

def run_phase1_lottery(win, outlet):
    """Show 5 face-down cards, let the user click one, reveal the name."""
    outlet.push_sample([MARKER_PHASE1_START])

    names = CONFIG["name_pool"][:]
    random.shuffle(names)                # random card-to-name mapping

    n_cards = len(names)
    card_w, card_h = 0.20, 0.30
    spacing = 0.05
    total_w = n_cards * card_w + (n_cards - 1) * spacing
    start_x = -total_w / 2 + card_w / 2
    positions = [(start_x + i * (card_w + spacing), -0.02) for i in range(n_cards)]

    # Build cards
    cards = []
    for pos in positions:
        face = visual.Rect(
            win, width=card_w, height=card_h, pos=pos,
            fillColor=CONFIG["color_card_face"],
            lineColor=CONFIG["color_card_border"], lineWidth=2.0,
        )
        glyph = visual.TextStim(
            win, text="?", pos=pos, height=0.10,
            color=CONFIG["color_accent"], font=CONFIG["font_display"], bold=True,
        )
        cards.append({"rect": face, "glyph": glyph, "pos": pos})

    instr = visual.TextStim(
        win,
        text="Choose a card.  Click it to claim your identity.",
        pos=(0, 0.30), height=0.034,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
    )

    mouse = event.Mouse(visible=True, win=win)
    mouse.clickReset()

    chosen_index = None
    pulse_t0 = core.getTime()

    while chosen_index is None:
        check_quit()
        t = core.getTime() - pulse_t0
        # subtle breathing glow on all cards
        pulse = 0.5 + 0.5 * np.sin(t * 2.0)

        draw_backdrop(win)
        header_bar(win, "PHASE 1 // IDENTITY DRAW")
        instr.draw()

        for i, card in enumerate(cards):
            hovered = card["rect"].contains(mouse)
            if hovered:
                card["rect"].lineColor = CONFIG["color_accent"]
                card["rect"].lineWidth = 3.5
                card["rect"].fillColor = "#1F2937"
            else:
                card["rect"].lineColor = CONFIG["color_card_border"]
                card["rect"].lineWidth = 2.0
                card["rect"].fillColor = CONFIG["color_card_face"]
            card["rect"].draw()
            card["glyph"].opacity = 0.55 + 0.35 * pulse if hovered else 0.7
            card["glyph"].draw()

        footer_hint(win, "Click a card with the mouse  ·  [ESC] to abort")
        win.flip()

        if mouse.getPressed()[0]:
            for i, card in enumerate(cards):
                if card["rect"].contains(mouse):
                    chosen_index = i
                    break
            # debounce
            while mouse.getPressed()[0]:
                core.wait(0.01)

    target_name = names[chosen_index]

    # ── Reveal animation ──────────────────────────────────────────────────
    reveal_card = cards[chosen_index]
    reveal_text = visual.TextStim(
        win, text=target_name, pos=reveal_card["pos"], height=0.055,
        color=CONFIG["color_accent_2"], font=CONFIG["font_display"], bold=True,
    )

    n_frames = 45
    for f in range(n_frames):
        progress = f / n_frames
        # collapse-then-expand flip effect on width
        scale = abs(np.cos(progress * np.pi))
        if scale < 0.02:
            scale = 0.02
        reveal_card["rect"].width = card_w * scale
        flipped = progress > 0.5

        draw_backdrop(win)
        header_bar(win, "PHASE 1 // IDENTITY DRAW")
        instr.draw()

        for i, card in enumerate(cards):
            if i == chosen_index:
                if flipped:
                    card["rect"].fillColor = CONFIG["color_panel"]
                    card["rect"].lineColor = CONFIG["color_accent_2"]
                    card["rect"].lineWidth = 3.5
                card["rect"].draw()
                if flipped:
                    reveal_text.opacity = (progress - 0.5) * 2
                    reveal_text.draw()
                else:
                    card["glyph"].opacity = 1 - progress * 2
                    card["glyph"].draw()
            else:
                card["rect"].opacity = 1 - progress * 0.6
                card["rect"].draw()
                card["glyph"].opacity = (1 - progress) * 0.7
                card["glyph"].draw()
                card["rect"].opacity = 1.0
                card["glyph"].opacity = 1.0

        win.flip()

    reveal_card["rect"].width = card_w

    # ── Memorization screen ──────────────────────────────────────────────
    mouse.setVisible(False)

    big_name = visual.TextStim(
        win, text=target_name, pos=(0, 0.12), height=0.16,
        color=CONFIG["color_accent_2"], font=CONFIG["font_display"], bold=True,
    )
    rule_text = (
        "Memorize this name. It is your secret.\n\n"
        "In the next phase, DENY KNOWING IT.\n"
        "Press the DOWN ARROW for ALL names — including your target —\n"
        "to simulate hiding the truth."
    )
    rule = visual.TextStim(
        win, text=rule_text, pos=(0, -0.18), height=0.032,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        wrapWidth=1.4, alignText="center",
    )
    box = visual.Rect(
        win, width=1.35, height=0.78, pos=(0, 0),
        fillColor=CONFIG["color_panel"],
        lineColor=CONFIG["color_accent_2"], lineWidth=1.5, opacity=0.55,
    )

    pulse_t0 = core.getTime()
    while True:
        check_quit()
        keys = event.getKeys(keyList=["space"])
        if keys:
            break
        t = core.getTime() - pulse_t0
        big_name.opacity = 0.75 + 0.25 * np.sin(t * 3.0)

        draw_backdrop(win)
        header_bar(win, "PHASE 1 // IDENTITY CONFIRMED")
        box.draw()
        visual.TextStim(
            win, text="YOUR SECRET", pos=(0, 0.30), height=0.026,
            color=CONFIG["color_dim"], font=CONFIG["font_body"],
        ).draw()
        big_name.draw()
        rule.draw()
        footer_hint(win, "Press [SPACE] when you have memorized it")
        win.flip()

    return target_name


# =============================================================================
# Phase 2 — CIT oddball loop
# =============================================================================

def build_trial_sequence(target_name, n_trials, target_ratio):
    """Pseudo-randomized order, never two probes in a row."""
    n_targets = max(1, int(round(n_trials * target_ratio)))
    n_irrelevant_total = n_trials - n_targets
    irrelevants = [n for n in CONFIG["name_pool"] if n != target_name]

    # roughly balanced irrelevant pool
    per_irr = n_irrelevant_total // len(irrelevants)
    remainder = n_irrelevant_total - per_irr * len(irrelevants)
    irr_pool = []
    for i, name in enumerate(irrelevants):
        count = per_irr + (1 if i < remainder else 0)
        irr_pool.extend([(name, "irrelevant")] * count)

    target_pool = [(target_name, "probe")] * n_targets
    all_trials = irr_pool + target_pool

    # Pseudo-randomize: no two probes back-to-back
    for _ in range(500):
        random.shuffle(all_trials)
        ok = all(
            not (all_trials[i][1] == "probe" and all_trials[i + 1][1] == "probe")
            for i in range(len(all_trials) - 1)
        )
        if ok:
            break
    return all_trials


def show_phase2_intro(win):
    box = visual.Rect(
        win, width=1.2, height=0.55, pos=(0, 0),
        fillColor=CONFIG["color_panel"],
        lineColor=CONFIG["color_accent"], lineWidth=1.5, opacity=0.55,
    )
    title = visual.TextStim(
        win, text="PHASE 2 — INTERROGATION",
        pos=(0, 0.16), height=0.05,
        color=CONFIG["color_accent"], font=CONFIG["font_display"], bold=True,
    )
    body = visual.TextStim(
        win,
        text=(
            "Names will appear one at a time.\n"
            "Press the DOWN ARROW for every name.\n\n"
            "Keep your eyes on the cross between names.\n"
            "Stay still. Breathe normally."
        ),
        pos=(0, -0.05), height=0.030,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        wrapWidth=1.1, alignText="center",
    )
    while True:
        draw_backdrop(win)
        header_bar(win, "PHASE 2 // BEGIN INTERROGATION")
        box.draw()
        title.draw()
        body.draw()
        footer_hint(win, "Press [SPACE] to begin  ·  [ESC] to abort")
        win.flip()
        wait_for_keys(["space"])
        return


def stylized_fixation(win):
    """Neon cross with a faint surrounding ring."""
    visual.Circle(
        win, radius=0.045, pos=(0, 0),
        lineColor=CONFIG["color_panel"], lineWidth=1.5,
        fillColor=None,
    ).draw()
    visual.TextStim(
        win, text="+", pos=(0, 0), height=0.07,
        color=CONFIG["color_fixation"], font=CONFIG["font_display"], bold=True,
    ).draw()


def run_phase2_cit(win, outlet, target_name):
    outlet.push_sample([MARKER_PHASE2_START])

    trials = build_trial_sequence(
        target_name, CONFIG["n_trials"], CONFIG["target_ratio"]
    )

    fixation_clock = core.Clock()
    rt_clock = core.Clock()
    results = []

    stim_text = visual.TextStim(
        win, text="", pos=(0, 0), height=0.13,
        color=CONFIG["color_text"], font=CONFIG["font_display"], bold=True,
    )

    for trial_idx, (name, trial_type) in enumerate(trials, start=1):
        check_quit()

        # ── Fixation ────────────────────────────────────────────────────
        fix_dur = random.uniform(CONFIG["fixation_min"], CONFIG["fixation_max"])
        fixation_clock.reset()
        while fixation_clock.getTime() < fix_dur:
            stylized_fixation(win)
            win.flip()
            check_quit()

        # ── Stimulus (push LSL marker on the exact flip) ────────────────
        stim_text.text = name
        marker_code = MARKER_PROBE if trial_type == "probe" else MARKER_IRRELEVANT

        event.clearEvents()
        stim_text.draw()
        # Register a one-shot callback so the marker is pushed at the moment
        # the flip is committed to the screen, not before.
        win.callOnFlip(outlet.push_sample, [marker_code])
        win.callOnFlip(rt_clock.reset)
        flip_time = win.flip()
        lsl_stamp = local_clock()

        # ── Response collection ─────────────────────────────────────────
        response = None
        rt = None
        stim_offset_time = flip_time + CONFIG["stimulus_duration"]
        deadline = flip_time + CONFIG["response_window"]

        stim_visible = True
        while core.getTime() < deadline:
            # Clear stimulus after its display duration; keep listening for RT.
            if stim_visible and core.getTime() >= stim_offset_time:
                win.flip()
                stim_visible = False
            elif stim_visible:
                stim_text.draw()
                win.flip()

            keys = event.getKeys(
                keyList=[CONFIG["response_key"], CONFIG["quit_key"]],
                timeStamped=rt_clock,
            )
            if keys:
                key_name, key_time = keys[0]
                if key_name == CONFIG["quit_key"]:
                    raise KeyboardInterrupt("User pressed escape.")
                response = key_name
                rt = key_time
                break

        # Make sure stim is off before ITI
        if stim_visible:
            win.flip()

        # ── ITI ────────────────────────────────────────────────────────
        iti_clock = core.Clock()
        while iti_clock.getTime() < CONFIG["iti"]:
            win.flip()
            check_quit()

        results.append({
            "trial": trial_idx,
            "stimulus": name,
            "trial_type": trial_type,
            "marker_code": marker_code,
            "response": response if response is not None else "none",
            "rt_seconds": round(rt, 4) if rt is not None else "",
            "stim_onset_psychopy": round(flip_time, 5),
            "stim_onset_lsl": round(lsl_stamp, 5),
        })

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
        "trial", "stimulus", "trial_type", "marker_code",
        "response", "rt_seconds",
        "stim_onset_psychopy", "stim_onset_lsl",
        "target_name",
    ]
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row_out = dict(row)
            row_out["target_name"] = target_name
            writer.writerow(row_out)
    return fname


def show_goodbye(win, csv_path, n_trials):
    box = visual.Rect(
        win, width=1.25, height=0.55, pos=(0, 0),
        fillColor=CONFIG["color_panel"],
        lineColor=CONFIG["color_accent"], lineWidth=1.5, opacity=0.55,
    )
    title = visual.TextStim(
        win, text="CASE CLOSED",
        pos=(0, 0.16), height=0.06,
        color=CONFIG["color_accent"], font=CONFIG["font_display"], bold=True,
    )
    body = visual.TextStim(
        win,
        text=(
            f"{n_trials} trials recorded.\n\n"
            f"Behavioral data saved to:\n{os.path.basename(csv_path)}\n\n"
            "Thank you for your testimony."
        ),
        pos=(0, -0.06), height=0.028,
        color=CONFIG["color_text"], font=CONFIG["font_body"],
        wrapWidth=1.15, alignText="center",
    )
    draw_backdrop(win)
    header_bar(win, "EXPERIMENT COMPLETE")
    box.draw()
    title.draw()
    body.draw()
    footer_hint(win, "Press any key to exit")
    win.flip()
    event.waitKeys()


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed()
    outlet = init_lsl_outlet()
    # Give consumers a moment to subscribe to the stream.
    core.wait(0.5)

    win = make_window()
    results = []
    target_name = None
    csv_path = None

    try:
        show_intro(win)
        target_name = run_phase1_lottery(win, outlet)
        show_phase2_intro(win)
        results = run_phase2_cit(win, outlet, target_name)
        outlet.push_sample([MARKER_EXPERIMENT_END])
        csv_path = save_results(results, target_name)
        show_goodbye(win, csv_path, len(results))

    except KeyboardInterrupt:
        # graceful abort — still try to save whatever we have
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
