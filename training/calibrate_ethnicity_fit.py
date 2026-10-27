from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _current_multipliers() -> dict[str, dict[str, float]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "engine_dash", str(Path(__file__).resolve().parent.parent / "backend" / "engine.py")
    )
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return {k: dict(v) for k, v in engine.ETHNICITY_FIT.items()}


def fit(
    existing: dict[str, dict[str, float]], measurements: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    updated = {}
    for ethnicity, m in measurements.items():
        base = dict(existing.get(ethnicity, {"w": 1.0, "h": 1.0, "shoulder": 1.0}))
        for key in ("w", "h", "shoulder"):
            if key in m:
                base[key] = float(m[key])
        updated[ethnicity] = base
    for ethnicity, base in existing.items():
        updated.setdefault(ethnicity, dict(base))
    return updated


def _load_csv(path: Path) -> dict[str, dict[str, float]]:
    import csv

    out: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            eth = row["ethnicity"].strip()
            if not eth:
                continue
            entry = {}
            for key in ("w", "h", "shoulder"):
                raw = row.get(key, "").strip()
                if raw:
                    entry[key] = float(raw)
            if entry:
                out[eth] = entry
    return out


def _render(updated: dict[str, dict[str, float]]) -> str:
    lines = ["ETHNICITY_FIT = {"]
    for eth in sorted(updated):
        v = updated[eth]
        lines.append(f'    "{eth}": dict(w={v["w"]:.3g}, h={v["h"]:.3g}, shoulder={v["shoulder"]:.3g}),')
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate ETHNICITY_FIT multipliers from measured ratios")
    parser.add_argument("--measurements", type=Path, help="CSV with columns ethnicity,w,h,shoulder")
    parser.add_argument("--diff", action="store_true", help="only print entries that differ from current")
    args = parser.parse_args()

    existing = _current_multipliers()
    n_before = len(existing)

    if args.measurements is None or not args.measurements.exists():
        print("no measurements file; printing current multipliers")
        print(_render(existing))
        return 0 if args.measurements is None else 1

    measured = _load_csv(args.measurements)
    if not measured:
        print(f"no usable rows in {args.measurements}")
        return 1

    updated = fit(existing, measured)

    sanity = []
    for eth, v in updated.items():
        if v["w"] < 0.8 or v["w"] > 1.3 or v["shoulder"] < 0.8 or v["shoulder"] > 1.3:
            sanity.append(f"  {eth}: out of range w={v['w']:.3f} shoulder={v['shoulder']:.3f}")
    if sanity:
        print("warning: values outside plausible range (overridden):")
        print("\n".join(sanity))

    if args.diff:
        changed = {
            k: f"w={v['w']:.3g}, h={v['h']:.3g}, shoulder={v['shoulder']:.3g}"
            for k, v in updated.items()
            if existing.get(k) != v
        }
        if not changed:
            print("no changes from measurements")
            return 0
        print("updated entries:")
        for k, v in changed.items():
            print(f'  "{k}": dict({v}),')
    else:
        print(_render(updated))

    n_changed = sum(1 for k in updated if existing.get(k) != updated[k])
    print(f"calibrated {n_changed}/{n_before} ethnicities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
