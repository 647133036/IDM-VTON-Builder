import threading
import time
import traceback
from pathlib import Path

from engine import compose_tryon, guess_audience, guess_garment_type
from store import BASES, RESULTS, UPLOADS, get_job, list_jobs, patch_job, settings as load_settings

_stop = threading.Event()


def _base_file(base: dict) -> Path:
    if base.get("path"):
        return Path(base["path"])
    return BASES / base["file"]


def process_job(job: dict, bases: list[dict]) -> None:
    patch_job(job["id"], {"status": "running", "progress": 8, "message": "读取底图与服装"})
    base = next((b for b in bases if b["id"] == job["base_id"]), None)
    if not base:
        raise RuntimeError("未找到指定模特底座")
    src = UPLOADS / Path(job["source_url"]).name
    if not src.exists():
        src = UPLOADS / job["source_name"]
    if not src.exists():
        raise RuntimeError("服装原图不存在")
    base_path = _base_file(base)
    if not base_path.exists():
        raise RuntimeError("模特底图文件缺失，请先上传底座")

    audience = job.get("audience") or "auto"
    if audience == "auto":
        audience = guess_audience(job.get("source_name", ""), base.get("audience", "women"))
    garment_type = job.get("garment_type") or "auto"
    if garment_type == "auto":
        garment_type = guess_garment_type(job.get("source_name", ""))

    patch_job(job["id"], {"progress": 28, "message": "分割服装并估计垂感", "audience": audience, "garment_type": garment_type})
    cfg = load_settings()
    long_side = int(cfg.get("output_long_side", 2048))
    background = job.get("background") or cfg.get("background", "white")
    result = compose_tryon(base_path, src, audience, garment_type, long_side, background)
    patch_job(job["id"], {"progress": 78, "message": "输出成品图"})

    ext = "png" if job.get("output_format") == "png" or background == "transparent" else "jpg"
    out_path = RESULTS / f"{job['id']}.{ext}"
    if ext == "png":
        result.save(out_path, "PNG")
    else:
        result.convert("RGB").save(out_path, "JPEG", quality=94, optimize=True)
    preview = RESULTS / f"{job['id']}_preview.jpg"
    thumb = result.convert("RGB")
    thumb.thumbnail((640, 860))
    thumb.save(preview, "JPEG", quality=88)
    patch_job(
        job["id"],
        {
            "status": "done",
            "progress": 100,
            "message": "已完成，可下载或推送到商品页",
            "result_url": f"/files/results/{out_path.name}",
            "preview_url": f"/files/results/{preview.name}",
        },
    )


def loop() -> None:
    job = None
    while not _stop.is_set():
        try:
            jobs = list_jobs()
            queued = [j for j in jobs if j["status"] == "queued"]
            if not queued:
                time.sleep(0.6)
                continue
            job = queued[-1]
            from store import load_state

            bases = load_state()["bases"]
            process_job(job, bases)
        except Exception as exc:
            try:
                if job:
                    patch_job(
                        job["id"],
                        {
                            "status": "failed",
                            "progress": 100,
                            "message": "生成失败",
                            "error": str(exc),
                        },
                    )
            except Exception:
                traceback.print_exc()
            time.sleep(0.4)


def start_worker() -> threading.Thread:
    t = threading.Thread(target=loop, name="tryon-worker", daemon=True)
    t.start()
    return t
