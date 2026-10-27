import threading
import time
import traceback
from pathlib import Path

from engine import compose_tryon, engine_status, guess_audience, guess_garment_type
from store import BASES, RESULTS, UPLOADS, claim_next_job, find_in_dir, patch_job
from store import settings as load_settings


def _default_workers() -> int:
    from os import cpu_count, getenv

    return max(1, min(int(getenv("WORKERS", "4")), cpu_count() or 4))


def _base_file(base: dict) -> Path:
    if base.get("path"):
        return Path(base["path"])
    return BASES / base["file"]


def process_job(job: dict, bases: list[dict]) -> None:
    patch_job(job["id"], {"progress": 8, "message": "Reading base and garment"})
    base = next((b for b in bases if b["id"] == job["base_id"]), None)
    if not base:
        raise RuntimeError("Base model not found")
    src = UPLOADS / Path(job["source_url"]).name if job.get("source_url") else None
    if not (src and src.exists()):
        src = UPLOADS / job["source_name"] if job.get("source_name") else None
    if not (src and src.exists()):
        name = Path(job.get("source_url") or job.get("source_name", "")).name
        src = find_in_dir(UPLOADS, name)
    if not (src and src.exists()):
        raise RuntimeError("Garment source image missing")
    base_path = _base_file(base)
    if not base_path.exists():
        raise RuntimeError("Base model image missing; upload a base first")

    audience = job.get("audience") or "auto"
    if audience == "auto":
        audience = base.get("audience") or guess_audience(job.get("source_name", ""), "women")
    garment_type = job.get("garment_type") or "auto"
    if garment_type == "auto":
        garment_type = guess_garment_type(job.get("source_name", ""))

    patch_job(
        job["id"],
        {
            "progress": 28,
            "message": "Segmenting garment and estimating drape",
            "audience": audience,
            "garment_type": garment_type,
        },
    )
    cfg = load_settings()
    long_side = int(cfg.get("output_long_side", 2048))
    background = job.get("background") or cfg.get("background", "white")
    engine_info = engine_status()
    engine_label = (
        "IDM-VTON diffusion engine" if engine_info["idm_vton_available"] else "composite engine (no-GPU fallback)"
    )
    patch_job(job["id"], {"progress": 35, "message": f"Calling {engine_label}"})
    ethnicity = base.get("ethnicity") or ""
    result = compose_tryon(base_path, src, audience, garment_type, long_side, background, ethnicity)
    patch_job(job["id"], {"progress": 78, "message": "Rendering final image"})

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
            "message": "Done; download or push to product page",
            "result_url": f"/files/results/{out_path.name}",
            "preview_url": f"/files/results/{preview.name}",
        },
    )


def _worker_loop() -> None:
    while True:
        job = claim_next_job()
        if job is None:
            time.sleep(0.4)
            continue
        try:
            from store import load_state

            bases = load_state()["bases"]
            process_job(job, bases)
        except Exception as exc:  # noqa: BLE001
            try:
                patch_job(
                    job["id"],
                    {
                        "status": "failed",
                        "progress": 100,
                        "message": "Generation failed",
                        "error": str(exc),
                    },
                )
            except Exception:  # noqa: BLE001
                traceback.print_exc()


class WorkerPool:
    def __init__(self, workers: int | None = None) -> None:
        self._workers = max(1, int(workers if workers is not None else _default_workers()))
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for i in range(self._workers):
            t = threading.Thread(target=_worker_loop, name=f"tryon-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def join(self) -> None:
        for t in self._threads:
            t.join(timeout=0.1)


def start_worker(workers: int | None = None) -> WorkerPool:
    pool = WorkerPool(workers if workers is not None else _default_workers())
    pool.start()
    return pool
