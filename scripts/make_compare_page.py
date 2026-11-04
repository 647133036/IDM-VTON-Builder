#!/usr/bin/env python3
"""Build a login-free before/after comparison page (garment | person | try-on).

Reads the run dir's report.json (written by run_tryon_batch.py) plus the
source folders, emits a single self-contained index.html using relative
image paths — open it in any browser or serve it with `python -m http.server`.

Usage:
  python scripts/make_compare_page.py \
      --runs  /mnt/workspace/vton/market_analysis/vton_runs \
      --out   /mnt/workspace/vton/market_analysis/vton_runs/../vton_compare
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = args.runs.resolve()
    report_path = runs / "report.json"
    if not report_path.exists():
        print(f"ERROR: no report.json in {runs} (run run_tryon_batch.py first)")
        sys.exit(1)
    report = json.loads(report_path.read_text())

    out = (args.out or (runs.parent / "vton_compare")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import os

    def rel(p: Path) -> str:
        return os.path.relpath(str(Path(p).resolve()), start=str(out))

    rows = []
    g_dir = runs.parent / "vton_flatlays"
    p_dir = runs.parent / "vton_models"
    for i, r in enumerate(report):
        g, p = r.get("garment"), r.get("person")
        o = r.get("out")
        o_path = runs / o if o else None
        degraded = " ⚠degraded" if r.get("degraded") else ""
        rows.append(f"""
    <div class="card">
      <div class="label">Pair {r.get('pair', i)}{degraded} &nbsp; <span class="meta">{(r.get('prompt') or '')[:120]}</span></div>
      <div class="cols">
        <figure><div class="cap">garment</div><img src="{rel(g_dir / g) if g else ''}"></figure>
        <figure><div class="cap">person</div><img src="{rel(p_dir / p) if p else ''}"></figure>
        <figure><div class="cap">try-on result</div><img src="{rel(o_path) if o_path else ''}"></figure>
      </div>
    </div>""")

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>VTON try-on results</title><style>
body{{font-family:system-ui;background:#0f1115;color:#e8eaed;margin:24px}}
h1{{font-size:20px}} .meta{{color:#8b93a1;font-size:12px}}
.card{{background:#171a21;border-radius:10px;padding:14px;margin-bottom:18px}}
.cols{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}}
figure{{margin:0}} img{{width:100%;border-radius:8px;background:#000;min-height:200px;object-fit:contain}}
.cap{{font-size:12px;color:#8b93a1;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}}
.note{{color:#8b93a1;font-size:13px;margin-bottom:16px}}
</style></head><body>
<h1>VTON try-on comparison</h1>
<p class="note">garment (flat-lay) | person (model photo) | try-on result — 共 {len(report)} 对。
生成于 {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}。判断标准:衣服必须"穿"在模特身上(垂坠/褶皱随身材),而非平铺贴图。</p>
{''.join(rows)}
</body></html>"""
    dest = out / "index.html"
    dest.write_text(html)
    print(f"[compare] wrote {dest} ({len(report)} pairs)")


if __name__ == "__main__":
    main()
