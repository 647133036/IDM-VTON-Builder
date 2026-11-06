#!/usr/bin/env python3
"""bench_run.py — W1 基准测试驱动器(06_执行计划.md 第 1 周交付物)。

用法:
    python3 scripts/bench_run.py check                 # 校验 input/ + lines.json
    python3 scripts/bench_run.py submit                 # 对每个 atelier 线批量提交 50 图
    python3 scripts/bench_run.py wait                   # 轮询 job 完成并下载结果
    python3 scripts/bench_run.py sheet                  # 生成评分表 CSV + Markdown 模板
    python3 scripts/bench_run.py score                  # 汇总已填写的评分表 -> 摘要 + LoRA 优先级
    python3 scripts/bench_run.py page                   # 生成 benchmark 对比页 HTML

设计(与 06_执行计划 对齐):
    - 输入:  benchmark/input/ 下 50 张平铺图, 文件名 <序号>_<品类>_<变体>.jpg,
      品类 ∈ basic/dress/kids/outerwear(基础款 15 / 连衣裙 15 / 童装 10 / 外套 10),
      变体 solid=纯色 pattern=花纹 等。audience/garment_type 优先取 manifest.json,
      缺省按品类映射(与 engine 的猜测规则一致)。
    - 线:  benchmark/lines.json。type=atelier 走本服务 API(每个引擎一个实例/端口);
      type=manual 表示外部 API(FASHN/可灵), 结果人工放进 results/<line>/<title>.jpg。
      引擎内部已统一 seed=42 / 30 步; 2048 长边与白底来自 settings(默认值), 提交前请确认。
    - 记录:  benchmark/runs/<line>.jsonl 逐 job 落账, submit 幂等(跳过已 done 的 title)。

引擎线实例配置(每台一端口, 见 benchmark/README.md):
    idm-vton     : LICENSE_MODE=research
    sdxl-lora    : LICENSE_MODE=commercial (默认加载 models/sdxl/lora)
    sdxl-no-lora : LICENSE_MODE=commercial + 把 models/sdxl/lora 改名 lora_off 后重启

纯标准库, 无第三方依赖。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_BATCH_FILES = 50  # 服务端上限; 06 计划按 25 一批提交, 留足体积余量
DEFAULT_CHUNK = 25
POLL_SECONDS = 5
WAIT_TIMEOUT = 3600  # 单线 50 张的最长等待(秒)
DIMENSIONS = ["fit", "pattern", "detail", "body", "fabric"]  # 各 20 分
PASS_LINE = 80  # 两人均分 >= 80 为通过
# 品类 -> (audience, garment_type) 缺省映射
CATEGORY_DEFAULTS = {
    "basic": ("women", "auto"),
    "dress": ("women", "dress"),
    "kids": ("girls", "auto"),
    "outer": ("women", "auto"),
}
NAME_RE = re.compile(r"^(\d+)_(\w+?)_(.*)\.(jpg|jpeg|png|webp)$", re.I)


class BenchError(RuntimeError):
    pass


def bench_root(args) -> Path:
    if getattr(args, "root", None):
        return Path(args.root).resolve()
    b = REPO_ROOT / "benchmark"
    return b


def bench_dir(args) -> Path:
    root = bench_root(args)
    p = root / "benchmark" if root.name != "benchmark" else root
    return p


def load_lines(args) -> list[dict]:
    p = bench_dir(args) / "lines.json"
    if not p.is_file():
        raise BenchError(f"缺少 {p} — 先建 lines.json(模板见 benchmark/README.md)")
    lines = json.loads(p.read_text())
    if not isinstance(lines, list) or not lines:
        raise BenchError("lines.json 必须是非空数组")
    for i, ln in enumerate(lines):
        for key in ("id", "label", "type"):
            if key not in ln:
                raise BenchError(f"lines.json 第 {i + 1} 条缺字段 {key}")
        if ln["type"] == "atelier" and not ln.get("url"):
            raise BenchError(f"line {ln['id']}: type=atelier 必须给 url")
    return lines


def load_inputs(args) -> list[dict]:
    """扫描 benchmark/input/, 返回 [{file,title,category,variant,audience,garment_type,pattern}]。"""
    bdir = bench_dir(args)
    in_dir = bdir / "input"
    manifest_path = bdir / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        for m in json.loads(manifest_path.read_text()):
            manifest[str(m.get("file", "")).lower()] = m
    items = []
    if in_dir.is_dir():
        for f in sorted(in_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            m = manifest.get(f.name.lower(), {})
            mo = NAME_RE.match(f.name)
            if mo:
                idx, category, variant = mo.group(1), mo.group(2).lower(), mo.group(3).lower()
            else:
                idx, category, variant = "00", "basic", ""
            cat_default = CATEGORY_DEFAULTS.get(category, ("women", "auto"))
            items.append(
                {
                    "file": f.name,
                    "title": f.stem,
                    "idx": int(idx) if idx.isdigit() else 0,
                    "category": category,
                    "variant": variant,
                    "audience": str(m.get("audience", cat_default[0])),
                    "garment_type": str(m.get("garment_type", cat_default[1])),
                    "pattern": 1 if (m.get("pattern") or "pattern" in variant) else 0,
                }
            )
    items.sort(key=lambda x: (x["idx"], x["file"]))
    return items


def http_json(
    url: str,
    token: str | None = None,
    data: bytes | None = None,
    method: str | None = None,
    content_type: str | None = None,
    extra_headers: dict | None = None,
    timeout: int = 60,
):
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None and content_type:
        req.add_header("Content-Type", content_type)
    for k, v in (extra_headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raise BenchError(f"HTTP {e.code} from {url}: {e.read()[:300]!r}") from None


def login(base_url: str, username: str, password: str) -> str:
    body = json.dumps({"username": username, "password": password}).encode()
    res = http_json(f"{base_url.rstrip('/')}/api/auth/login", data=body, method="POST", content_type="application/json")
    return res["token"]


def multipart(fields: dict, files: list[tuple[bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for k, v in fields.items():
        buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for data, filename in files:
        buf.write(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
        )
        buf.write(data)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def resolve_base(token: str, url: str, spec: str) -> dict:
    bases = http_json(f"{url.rstrip('/')}/api/bases", token)
    for b in bases:
        if b.get("active") is False:
            continue
        if spec and (str(b.get("id")) == spec or b.get("name") == spec):
            return b
    if not bases:
        raise BenchError(f"{url} 没有可用 base model, 先上传模特图")
    return bases[0]


def run_records_path(bdir: Path, line_id: str) -> Path:
    return bdir / "runs" / f"{line_id}.jsonl"


def read_runs(bdir: Path, line_id: str) -> list[dict]:
    p = run_records_path(bdir, line_id)
    if not p.is_file():
        return []
    out = []
    for i, line in enumerate(p.read_text().splitlines()):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                raise BenchError(f"{p} 第 {i + 1} 行不是合法 JSON") from None
    return out


def write_runs(bdir: Path, line_id: str, records: list[dict]) -> None:
    p = run_records_path(bdir, line_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""))


def cmd_check(args) -> int:
    lines = load_lines(args)
    inputs = load_inputs(args)
    print(f"lines.json: {len(lines)} 条线")
    for ln in lines:
        note = ln.get("url") if ln["type"] == "atelier" else f"(manual: 结果放入 results/{ln['id']}/)"
        print(f"  - {ln['id']:<12} {ln['type']:<7} {ln.get('label', ''):<20} {note}")
    print(f"input/: {len(inputs)} 张 (06 计划要求 50)")
    by_cat: dict[str, int] = {}
    for it in inputs:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
    for c, n in sorted(by_cat.items()):
        print(f"  - {c}: {n}")
    if len(inputs) < 50:
        print("注意: 输入不足 50 张, 先补齐再跑基准(06: 基础 15/连衣裙 15/童装 10/外套 10)")
    ok = True
    for ln in lines:
        if ln["type"] != "atelier":
            continue
        pw_env = ln.get("password_env", "STORE_ADMIN_PASSWORD")
        if not os.environ.get(pw_env):
            print(f"警告: line {ln['id']} 的密码环境变量 {pw_env} 未设置, submit 会失败")
            ok = False
    print("check " + ("通过" if ok else "有警告"))
    return 0 if ok else 1


def cmd_submit(args) -> int:
    bdir = bench_dir(args)
    lines = [ln for ln in load_lines(args) if ln["type"] == "atelier"]
    inputs = load_inputs(args)
    if not inputs:
        raise BenchError("benchmark/input/ 没有可用图片")
    for ln in lines:
        url = ln["url"].rstrip("/")
        pw_env = ln.get("password_env", "STORE_ADMIN_PASSWORD")
        password = os.environ.get(pw_env, "")
        if not password:
            raise BenchError(f"line {ln['id']}: 环境变量 {pw_env} 未设置")
        token = login(url, ln.get("username", "admin"), password)
        base = resolve_base(token, url, ln.get("base", ""))
        records = read_runs(bdir, ln["id"])
        done_titles = {r["title"] for r in records if r.get("status") == "done"}
        todo = [it for it in inputs if it["title"] not in done_titles]
        print(f"[{ln['id']}] base={base.get('name') or base.get('id')} 待提交 {len(todo)}/{len(inputs)}")
        # 按 (audience, garment_type) 分组, 每组 <= 25 张/次
        groups: dict[tuple, list[dict]] = {}
        for it in todo:
            groups.setdefault((it["audience"], it["garment_type"]), []).append(it)
        for (audience, gtype), group in sorted(groups.items()):
            for i in range(0, len(group), DEFAULT_CHUNK):
                chunk = group[i : i + DEFAULT_CHUNK]
                files = []
                for it in chunk:
                    files.append((bdir.joinpath("input", it["file"]).read_bytes(), it["file"]))
                body, ctype = multipart(
                    {
                        "base_id": str(base["id"]),
                        "audience": audience,
                        "garment_type": gtype,
                        "output_format": "jpg",
                        "background": ln.get("background", "white"),
                    },
                    files,
                )
                res = http_json(
                    f"{url}/api/jobs/batch",
                    token,
                    data=body,
                    method="POST",
                    content_type=ctype,
                    extra_headers={"X-Requested-With": "Atelier-Bench"},
                )
                jobs = res.get("jobs", [])
                if len(jobs) != len(chunk):
                    raise BenchError(f"[{ln['id']}] 期望 {len(chunk)} 个 job, 服务端返回 {len(jobs)}: {res}")
                by_title = {j.get("title"): j for j in jobs}
                for it in chunk:
                    job = by_title.get(it["title"])
                    if job is None:
                        # 服务端 title 是文件名去扩展名; 大小写差异兜底
                        job = next((j for j in jobs if j.get("title", "").lower() == it["title"].lower()), None)
                    if job is None:
                        raise BenchError(f"[{ln['id']}] 没有名为 {it['title']} 的 job: {res}")
                    records.append(
                        {
                            "line": ln["id"],
                            "label": ln.get("label", ln["id"]),
                            "title": it["title"],
                            "file": it["file"],
                            "category": it["category"],
                            "variant": it["variant"],
                            "pattern": it["pattern"],
                            "audience": audience,
                            "garment_type": gtype,
                            "base": str(base["id"]),
                            "job_id": job["id"],
                            "url": url,
                            "status": "pending",
                            "submitted_at": int(time.time()),
                        }
                    )
                print(
                    f"  submitted {len(chunk)} ({audience}/{gtype}) -> "
                    + ", ".join(j["id"] for j in jobs[:3])
                    + (" ..." if len(jobs) > 3 else "")
                )
        write_runs(bdir, ln["id"], records)
    print("submit 完成; 跑 `wait` 等待生成并下载结果")
    return 0


def cmd_wait(args) -> int:
    bdir = bench_dir(args)
    for ln in load_lines(args):
        if ln["type"] != "atelier":
            continue
        url = ln["url"].rstrip("/")
        pw_env = ln.get("password_env", "STORE_ADMIN_PASSWORD")
        password = os.environ.get(pw_env, "")
        if not password:
            print(f"[{ln['id']}] 跳过: 环境变量 {pw_env} 未设置")
            continue
        token = login(url, ln.get("username", "admin"), password)
        records = read_runs(bdir, ln["id"])
        pending = [r for r in records if r.get("status") == "pending"]
        if not pending:
            print(f"[{ln['id']}] 无待处理 job")
            continue
        res_dir = bdir / "results" / ln["id"]
        res_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        still = list(pending)
        while still and time.time() - t0 < WAIT_TIMEOUT:
            for i, r in enumerate(still):
                job = http_json(f"{url}/api/jobs/{r['job_id']}", token)
                status = job.get("status")
                if status == "done":
                    rel = str(job.get("result_url", "")).lstrip("/")
                    raw = http_bytes(f"{url}/{rel}", token)
                    ext = Path(rel).suffix or ".jpg"
                    dest = res_dir / f"{r['title']}{ext}"
                    dest.write_bytes(raw)
                    r["status"] = "done"
                    r["result_path"] = str(dest.relative_to(bdir))
                    r["engine"] = job.get("engine", "")
                    r["engine_degraded"] = job.get("engine_degraded", False)
                    r["done_at"] = int(time.time())
                    still[i] = None
                elif status == "failed":
                    r["status"] = "failed"
                    r["error"] = str(job.get("error") or job.get("message"))
                    r["done_at"] = int(time.time())
                    still[i] = None
                else:
                    print(f"  [{ln['id']}] {r['title']} {status} {job.get('progress', 0)}% {job.get('message', '')}")
            still = [s for s in still if s is not None]
            if still:
                time.sleep(POLL_SECONDS)
        write_runs(bdir, ln["id"], records)
        if still:
            print(f"[{ln['id']}] 超时({WAIT_TIMEOUT}s)仍有 {len(still)} 个未完成, 稍后再跑 wait")
        done_n = sum(1 for r in records if r.get("status") == "done")
        fail_n = sum(1 for r in records if r.get("status") == "failed")
        print(f"[{ln['id']}] done {done_n}, failed {fail_n}")
    return 0


def http_bytes(url: str, token: str | None) -> bytes:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def cmd_sheet(args) -> int:
    bdir = bench_dir(args)
    lines = load_lines(args)
    inputs = load_inputs(args)
    if not inputs:
        raise BenchError("benchmark/input/ 没有可用图片, 无法生成评分表")
    out_csv = bdir / "scoring" / "sheet.csv"
    out_md = bdir / "scoring" / "sheet_template.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ["idx", "title", "category", "variant", "pattern", "line"]
        + [f"{r}_{d}" for d in DIMENSIONS for r in ("A", "B")]
        + ["A_total", "B_total", "avg", "pass", "notes"]
    )
    rows = []
    for it in inputs:
        for ln in lines:
            row = {
                "idx": it["idx"],
                "title": it["title"],
                "category": it["category"],
                "variant": it["variant"],
                "pattern": it["pattern"],
                "line": ln["id"],
            }
            for d in DIMENSIONS:
                row[f"A_{d}"] = ""
                row[f"B_{d}"] = ""
            row.update({"A_total": "", "B_total": "", "avg": "", "pass": "", "notes": ""})
            rows.append(row)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    lines_md = "\n".join(f"- {ln['label'] or ln['id']}" for ln in lines)
    out_md.write_text(
        f"""# 评分表填写说明({time.strftime("%Y-%m-%d")})

线:
{lines_md}

填法:
- 数据在 `sheet.csv`(50 图 x N 线, 每人每维 0-20 分, 共 100 分; 2 人独立盲评填 A/B, 算均值)。
- 维度: fit=贴合 | pattern=花纹/图案位移 | detail=袖口下摆细节 | body=身体结构 | fabric=面料质感。
- 盲评: 给评审的对比页不要显示"哪条线是哪条引擎"(page 加 --blind 参数生成)。
- 花纹款(pattern=1)额外记"花纹位移是否可见"(pattern 维 <=10 记为位移失败), 汇总时单列"花纹位移率"。
- 通过线: 两人均分 avg >= {PASS_LINE}。
- 填完回 `python3 scripts/bench_run.py score` 出摘要与 LoRA 优先排序。
"""
    )
    print(f"评分表已生成: {out_csv} ({len(rows)} 行)")
    print(f"说明: {out_md}")
    return 0


def _totals(r: dict, prefix: str) -> float | None:
    vals = [r.get(f"{prefix}_{d}", "") for d in DIMENSIONS]
    if any(v in ("", None) for v in vals):
        return None
    return sum(float(v) for v in vals)


def cmd_score(args) -> int:
    bdir = bench_dir(args)
    lines = load_lines(args)
    sheet = bdir / "scoring" / "sheet.csv"
    if not sheet.is_file():
        raise BenchError(f"先跑 `sheet` 生成 {sheet}")
    with sheet.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("title")]
    for r in rows:
        a, b = _totals(r, "A"), _totals(r, "B")
        r["A_total"], r["B_total"] = a or "", b or ""
        filled = [x for x in (a, b) if x is not None]
        if len(filled) == 2:
            r["avg"] = round((a + b) / 2, 1)
            r["pass"] = "yes" if (a + b) / 2 >= PASS_LINE else "no"
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    filled_n = sum(1 for r in rows if r.get("avg"))
    # 按线汇总
    summary_lines: list[str] = []
    by_line: dict[str, list[dict]] = {}
    for ln in lines:
        rs = [r for r in rows if r.get("line") == ln["id"] and r.get("avg")]
        by_line[ln["id"]] = rs
        if not rs:
            continue
        n = len(rs)
        avg = sum(float(r["avg"]) for r in rs) / n
        passed = sum(1 for r in rs if r.get("pass") == "yes")
        pat = [r for r in rs if r.get("pattern") in ("1", "True", "true")]
        pat_fail = sum(1 for r in pat if r.get("avg") and _dim_avg(r, "pattern") <= 10)
        s = [
            f"### {ln['id']} ({ln.get('label', '')})\n"
            f"- 已评 {n} 图, 均分 {avg:.1f}, 通过 {passed}/{n} ({passed / n:.0%})"
        ]
        if pat:
            s.append(f"- 花纹款 {len(pat)} 张, 花纹位移失败 {pat_fail} ({pat_fail / len(pat):.0%})")
        by_cat: dict[str, list[float]] = {}
        for r in rs:
            by_cat.setdefault(r["category"], []).append(float(r["avg"]))
        ranking = sorted(((c, sum(v) / len(v)) for c, v in by_cat.items()), key=lambda x: x[1])
        s.append("- 品类均分(低->高): " + " | ".join(f"{c} {v:.1f}" for c, v in ranking))
        summary_lines.append("\n".join(s))
    # LoRA 优先级: 看"品类"在内部线(带 lora_target 的线, 缺省=所有 atelier 线)上的弱项,
    # 而不是看外部竞品 API 差。外部线仅作效果参照, 不进入 LoRA 决策。
    lora_lines = [ln for ln in lines if ln["type"] == "atelier" and ln.get("lora_target") is not False]
    if not lora_lines:
        lora_lines = [ln for ln in lines if ln["type"] == "atelier"]
    lora_ids = {ln["id"] for ln in lora_lines}
    cat_scores: dict[str, list[float]] = {}
    for r in rows:
        if r.get("line") in lora_ids and r.get("avg"):
            cat_scores.setdefault(r["category"], []).append(float(r["avg"]))
    cat_priority = sorted(((c, sum(v) / len(v), len(v)) for c, v in cat_scores.items()), key=lambda x: x[1])
    # 每线整体均分(排序展示用, 外部线仅参照)
    line_scores = []
    for lid, rs in by_line.items():
        if rs:
            line_scores.append((lid, sum(float(r["avg"]) for r in rs) / len(rs)))
    line_scores.sort(key=lambda x: x[1])
    md = bdir / "scoring" / "summary.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lora_block = (
        "\n".join(f"{i + 1}. {c}: 均分 {v:.1f} ({n} 张)" for i, (c, v, n) in enumerate(cat_priority))
        if cat_priority
        else "(内部线还没有完成评分, 暂无法排 LoRA 优先级)"
    )
    md.write_text(
        f"""# W1 基准汇总({time.strftime("%Y-%m-%d %H:%M")})

填写进度: {filled_n}/{len(rows)} 行(需 2 人均填齐才算一行)

{chr(10).join(summary_lines) if summary_lines else "(还没有完成评分)"}

## LoRA 优先级(内部线按品类弱项排, 低的先补数据 + 重训)

"""
        + lora_block
        + """
> 说明: 只统计带 `lora_target`(缺省=所有 atelier 线)的品类均分, 外部竞品线不进 LoRA 决策。

## 各线整体均分(外部线仅参照)

"""
        + "\n".join(f"{i + 1}. {lid}: 均分 {v:.1f}" for i, (lid, v) in enumerate(line_scores))
        + "\n"
    )
    print(md.read_text())
    print(f"摘要已写入 {md}")
    return 0


def _dim_avg(r: dict, dim: str) -> float | None:
    vals = [r.get(f"A_{dim}", ""), r.get(f"B_{dim}", "")]
    vals = [float(v) for v in vals if v not in ("", None)]
    return sum(vals) / len(vals) if vals else None


def cmd_page(args) -> int:
    bdir = bench_dir(args)
    lines = load_lines(args)
    inputs = load_inputs(args)
    if not inputs:
        raise BenchError("benchmark/input/ 没有可用图片")
    page_dir = bdir / "page"
    page_dir.mkdir(parents=True, exist_ok=True)
    summary = bdir / "scoring" / "summary.md"
    summary_txt = summary.read_text() if summary.is_file() else "<i>评分汇总尚未生成(填完 sheet 后跑 score)</i>"
    selected = []
    for ln in lines:
        sel = ln.get("selected") or []
        if sel:
            selected.append({"line": ln["id"], "label": ln.get("label", ln["id"]), "titles": sel})

    def img_rel(base_dir: str, rel: str) -> str:
        return f"{base_dir}/{rel}"

    def cell(ln_id: str, it: dict) -> str:
        p = bdir / "results" / ln_id / it["title"]
        found = next(
            (x for x in p.parent.glob(f"{it['title']}*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
            None,
        )
        if found:
            return f'<img src="../results/{ln_id}/{found.name}" loading="lazy" alt="{it["title"]} {ln_id}">'
        return '<div class="missing">未生成</div>'

    cols = "".join(f"<th>{ln.get('label') or ln['id']}</th>" for ln in lines)
    body_rows = []
    for it in inputs:
        tds = "".join(cell(ln["id"], it) for ln in lines)
        body_rows.append(f"<tr><td class='t'>{it['title']}</td><td>{img_rel('../input', it['file'])}</td>{tds}</tr>")
    sel_html = ""
    if selected:
        blocks = []
        for s in selected:
            imgs = []
            for t in s["titles"]:
                found = next(
                    (
                        x
                        for x in (bdir / "results" / s["line"]).glob(f"{t}*")
                        if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    ),
                    None,
                )
                if found:
                    imgs.append(
                        f'<figure><img src="../results/{s["line"]}/{found.name}" alt="{t}">'
                        f"<figcaption>{s['label']} · {t}</figcaption></figure>"
                    )
            blocks.append(f"<div class='sel'><h3>{s['label']}</h3>{''.join(imgs)}</div>")
        sel_html = f"<section><h2>精选(发伙伴用)</h2>{''.join(blocks)}</section>"
    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atelier Try-On · W1 基准对比</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px; color: #222; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }}
  td img {{ max-width: 180px; width: 100%; display: block; border-radius: 6px; }}
  td.t {{ font-family: ui-monospace, monospace; font-size: 12px; white-space: nowrap; }}
  .missing {{ color: #999; font-size: 12px; padding: 20px 0; }}
  figure {{ display: inline-block; margin: 0 12px 0 0; }}
  figure img {{ max-width: 260px; border-radius: 8px; }}
  figcaption {{ font-size: 12px; color: #666; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>W1 基准对比({time.strftime("%Y-%m-%d")})</h1>
<p class="meta">统一参数: 2048 长边 · 白底 · 30 步 · seed 42(引擎内建)。行数 = 50 输入图。</p>
{sel_html}
<section>
<h2>全量对比</h2>
<table>
<thead><tr><th>输入图</th><th>原图</th>{cols}</tr></thead>
<tbody>{"".join(body_rows)}</tbody>
</table>
</section>
<section>
<h2>评分汇总</h2>
<pre>{summary_txt}</pre>
</section>
</body>
</html>
"""
    out = page_dir / "index.html"
    out.write_text(html)
    print(f"对比页已生成: {out}")
    print("盲评版: 把 lines.json 各线的 label 留空(删掉), 重新跑 page 即可")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="W1 基准驱动器")
    ap.add_argument("command", choices=["check", "submit", "wait", "sheet", "score", "page"])
    ap.add_argument("--root", help="benchmark/ 所在目录(默认: 仓库根下的 benchmark/)")
    args = ap.parse_args()
    if args.root:
        # --root 直接指向 benchmark 目录本身
        global bench_dir

        def _fixed(args):
            return Path(args.root).resolve()

        bench_dir = _fixed  # type: ignore
    try:
        return {
            "check": cmd_check,
            "submit": cmd_submit,
            "wait": cmd_wait,
            "sheet": cmd_sheet,
            "score": cmd_score,
            "page": cmd_page,
        }[args.command](args)
    except BenchError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
