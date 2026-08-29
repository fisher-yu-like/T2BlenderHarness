"""One-shot provider-side codegen improvements from round-1 attempt-1 evidence."""

from pathlib import Path

p = Path("scripts/author_camera_case_source.py")
src = p.read_text(encoding="utf-8")
steps = []


def replace(name: str, old: str, new: str, required: bool = True) -> None:
    global src
    if old not in src:
        steps.append((name, "MISS", old[:60]))
        if required:
            raise SystemExit(f"pattern missing: {name}: {old[:80]}")
        return
    src = src.replace(old, new, 1)
    steps.append((name, "ok", ""))


# 1. Per-subject base offsets for stage alignment.
bases = {
    '"Garden"': -1.0,
    '"Pyramid"': 0.0,
    '"Mount Fuji"': 0.0,
    '"Blue Lagoon"': 0.0,
    '"Table"': -0.79,
    '"Alhambra"': 0.0,
    '"Vase"': -0.05,
    '"Burj Khalifa"': 0.0,
    '"Machu Picchu"': 0.0,
    '"Forbidden City"': 0.0,
    '"Laptop"': 0.0,
    '"Watch"': -0.055,
}
table = "SUBJECT_BASE_Z = {\n" + "\n".join(f"    {k}: {v}," for k, v in bases.items()) + "\n}\n\nPALETTES = {"
replace("base-z table", "PALETTES = {", table)

# 2. Stage slab top follows the plan staging height.
replace(
    "stage alignment",
    "    stage = mesh_object(\"support_surface\", rounded_box((0.0, 0.0, 0.0), (3.8, 2.8, 0.18), 0.06), environment)\n"
    "    stage.location = (subject_anchor[0], subject_anchor[1], 0.09)",
    "    # The staging surface rises to meet the plan's prop height so the\n"
    "    # subject base is supported instead of hovering (attempt-1 evidence).\n"
    "    stage_top = max(0.18, round(subject_anchor[2] + SUBJECT_BASE_Z.get(__SUBJECT_LABEL_LITERAL__, 0.0), 3))\n"
    "    stage = mesh_object(\"support_surface\", rounded_box((0.0, 0.0, 0.0), (3.8, 2.8, 0.18), 0.06), environment)\n"
    "    stage.location = (subject_anchor[0], subject_anchor[1], stage_top - 0.09)",
)

# 3. Orbit: tighter radius keeps the subject framed mid-orbit.
replace("orbit radius", '"        6.4,\\n"', '"        5.0,\\n"')

# 4. Pan swing reduced so the subject stays near frame.
replace("pan swing", '"    swing = 2.6\\n"', '"    swing = 1.5\\n"')

# 5. Zoom-in stops before overshoot.
replace(
    "zoom-in end",
    '"    near = (subject_center[0] + 1.0, subject_center[1] - 4.4, subject_center[2] + 1.9)\\n"',
    '"    near = (subject_center[0] + 1.4, subject_center[1] - 5.8, subject_center[2] + 2.3)\\n"',
)

# 6. Tilt-up ends lower for tall subjects.
replace(
    "tilt-up end",
    '"    high = (subject_center[0] + 1.8, subject_center[1] - 6.6, subject_center[2] + 5.8)\\n"',
    '"    high = (subject_center[0] + 2.0, subject_center[1] - 7.4, subject_center[2] + 4.2)\\n"',
)

# 7. First-person dolly keeps a stand-off offset (no facade collision).
replace(
    "first-person stand-off",
    '"        (0.0, -3.2, subject_center[2] + 0.4),\\n"',
    '"        (0.0, -5.6, subject_center[2] + 1.2),\\n"',
)

p.write_text(src, encoding="utf-8")
for name, status, _ in steps:
    print(name, status)
print("all improvements applied")
