"""Hand-laid SVG for the weekly scoring DAG (orthogonal edges, no crossings)."""

from pathlib import Path

FILL, STROKE, INK, EDGE, LABEL = "#eef1f4", "#8a94a3", "#1d2430", "#5b6575", "#1f4e8c"
FONT = "font-family='Arial, Helvetica, sans-serif'"


def box(x, y, w, h, *lines, bold=False):
    """Rounded task box with 1-2 centred lines."""
    cy = y + h / 2
    out = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='6' fill='{FILL}' stroke='{STROKE}'/>"]
    offs = [0] if len(lines) == 1 else [-8, 9]
    for t, o in zip(lines, offs):
        size, color = (13, INK) if o <= 0 else (11, EDGE)
        out.append(f"<text x='{x + w / 2}' y='{cy + o + 4}' text-anchor='middle' {FONT} "
                   f"font-size='{size}' fill='{color}'>{t}</text>")
    return "".join(out)


def diamond(cx, cy, w, h, *lines):
    """Branch operator drawn as a diamond."""
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    out = [f"<polygon points='{pts}' fill='#fff7e6' stroke='#c08a2b'/>"]
    for t, o in zip(lines, [-6, 10]):
        size, color = (12.5, INK) if o < 0 else (10.5, EDGE)
        out.append(f"<text x='{cx}' y='{cy + o + 4}' text-anchor='middle' {FONT} "
                   f"font-size='{size}' fill='{color}'>{t}</text>")
    return "".join(out)


def edge(d):
    """Orthogonal connector with arrow head."""
    return f"<path d='{d}' fill='none' stroke='{EDGE}' stroke-width='1.6' marker-end='url(#arr)'/>"


def label(x, y, t, anchor="middle"):
    """Edge label on a white pill so lines never run through text."""
    w = len(t) * 6.6 + 10
    x0 = x - w / 2 if anchor == "middle" else x
    tx = x if anchor == "middle" else x + w / 2
    return (f"<rect x='{x0}' y='{y - 11}' width='{w}' height='16' rx='8' fill='#ffffff' stroke='#dfe3e8'/>"
            f"<text x='{tx}' y='{y + 1}' text-anchor='middle' {FONT} font-size='11' fill='{LABEL}' "
            f"font-weight='bold'>{t}</text>")


W, H = 1060, 375
parts = [
    f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' style='max-width:100%;height:auto' "
    "role='img' aria-label='Weekly scoring DAG'>",
    "<defs><marker id='arr' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
    f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{EDGE}'/></marker></defs>",
    f"<rect width='{W}' height='{H}' fill='#ffffff'/>",
    # lane captions
    f"<text x='20' y='26' {FONT} font-size='11' fill='{EDGE}' letter-spacing='1'>MONITORING BRANCH</text>",
    f"<text x='20' y='170' {FONT} font-size='11' fill='{EDGE}' letter-spacing='1'>SCORING PATH</text>",
    f"<line x1='20' y1='140' x2='{W - 20}' y2='140' stroke='#e6e9ee' stroke-dasharray='4 4'/>",
    # edges first (drawn under nodes)
    edge("M150 230 H183"),                               # extract -> validate
    edge("M297 230 H358"),                               # validate -> build (valid)
    edge("M240 268 V340 H578"),                          # validate -> fallback (invalid)
    edge("M508 230 H578"),                               # build -> score
    edge("M738 230 H780 V268 H818"),                     # score -> assign
    edge("M738 340 H780 V302 H818"),                     # fallback -> assign
    edge("M433 210 V72 H546"),                           # build -> check_drift
    edge("M654 72 H700 V44 H778"),                       # drift -> retrain
    edge("M654 72 H700 V104 H778"),                      # drift -> no_drift
    # nodes
    box(20, 210, 130, 40, "extract_data"),
    diamond(240, 230, 114, 76, "validate_data", "branch"),
    box(358, 210, 150, 40, "build_features"),
    box(578, 206, 160, 48, "score_students", "model @champion"),
    box(578, 320, 160, 40, "fallback_rule", ),
    box(818, 258, 222, 56, "assign_tiers", "trigger: none_failed_min_one_success"),
    diamond(600, 72, 108, 72, "check_drift", "branch"),
    box(778, 26, 180, 36, "trigger_retraining"),
    box(778, 86, 180, 36, "no_drift"),
    # labels on clear segments
    label(327, 214, "valid"),
    label(410, 324, "invalid"),
    label(739, 34, "PSI > 0.2"),
    label(739, 120, "PSI ≤ 0.2"),
    "</svg>",
]
Path(__file__).with_name("scoring.svg").write_text("".join(parts))
print("scoring.svg written")


# ---------------- training DAG (same visual language) ----------------
TW, TH = 1060, 110
t = [
    f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {TW} {TH}' style='max-width:100%;height:auto' "
    "role='img' aria-label='Training DAG'>",
    "<defs><marker id='arr2' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
    f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{EDGE}'/></marker></defs>",
    f"<rect width='{TW}' height='{TH}' fill='#ffffff'/>",
]
for d in ("M210 55 H262", "M452 55 H504", "M754 55 H806"):
    t.append(f"<path d='{d}' fill='none' stroke='{EDGE}' stroke-width='1.6' marker-end='url(#arr2)'/>")
t += [
    box(20, 33, 190, 44, "snapshot_labelled_data", ),
    box(262, 33, 190, 44, "validate_training_data", "fails run if invalid"),
    box(504, 27, 250, 56, "train_and_track_experiments", "12 MLflow runs · gate · register"),
    box(806, 33, 234, 44, "await_human_approval", "promote alias champion"),
    "</svg>",
]
Path(__file__).with_name("training.svg").write_text("".join(t))
print("training.svg written")
