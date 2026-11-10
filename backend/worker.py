import os
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
    # 多 seed 出图: 印花/文字款 VTON_MULTI_SEED=3(默认), 纯色 =1。
    # 每个 seed 跑一次 compose_tryon, 按质检分排序取最高。
    # VTON_MULTI_SEED=0 强制单 seed; VTON_MULTI_SEED=1 也是单 seed。
    n_seeds = max(1, int(os.environ.get("VTON_MULTI_SEED", "3")))
    if n_seeds == 1:
        result = compose_tryon(base_path, src, audience, garment_type, long_side, background, ethnicity)
    else:
        from preserve_refit import refit_seeds_default, run_multiseed
        from quality_gate import score_for_sku

        seeds = refit_seeds_default()[:n_seeds]
        # 纯色款单 seed 控成本: job 未显式标 has_print=False 时按印花处理
        has_print = job.get("has_print", True)
        if not has_print:
            seeds = seeds[:1]

        def _task(seed):
            os.environ["VTON_SEED"] = str(seed)
            try:
                return compose_tryon(base_path, src, audience, garment_type, long_side, background, ethnicity)
            finally:
                os.environ.pop("VTON_SEED", None)

        def _scorer(img, seed):
            # 简化: 图案/纯色用 SSIM(全图中间 60% ROI); 文字类走 OCR(此处回退 SSIM)
            try:
                from PIL import Image

                ref_arr = Image.open(src).convert("RGB")
                out_arr = img.convert("RGB")
                if out_arr.size != ref_arr.size:
                    out_arr = out_arr.resize(ref_arr.size)
                h, w = ref_arr.size[1], ref_arr.size[0]
                box = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
                score, _tier = score_for_sku(ref_arr.crop(box), out_arr.crop(box), sku_has_text=False)
                return score
            except Exception:  # noqa: BLE001
                return 0.0

        result, _all_scores, _chosen = run_multiseed(_task, seeds, _scorer)
        result.info["multiseed"] = {"n_seeds": n_seeds, "seeds": seeds, "chosen_seed": _chosen}
        if len(seeds) > 1:
            patch_job(job["id"], {"message": f"Multi-seed ({n_seeds}) done, chosen seed={_chosen}"})
    engine_used = result.info.get("engine") or engine_info["engine"]
    degraded = bool(result.info.get("engine_degraded", False))
    degraded_reason = result.info.get("degraded_reason") or ""
    patch_job(
        job["id"],
        {
            "progress": 78,
            "message": "Rendering final image",
            "engine": engine_used,
            "engine_degraded": degraded,
            "degraded_reason": degraded_reason,
        },
    )
    if "multiseed" in result.info:
        patch_job(job["id"], {"multiseed": result.info["multiseed"]})

    ext = "png" if job.get("output_format") == "png" or background == "transparent" else "jpg"
    out_path = RESULTS / f"{job['id']}.{ext}"
    platform = job.get("platform", "none")
    if ext == "png" and platform == "none":
        result.save(out_path, "PNG")
    else:
        # 平台模板(shopify/amazon/aliexpress):1:1 居中裁剪 + 平台尺寸 + 纯白底 + JPEG 限体积。
        from platform_presets import PLATFORM_PRESETS, save_platform_result

        out_jpg = RESULTS / f"{job['id']}.jpg"
        if platform in PLATFORM_PRESETS:
            _, nbytes = save_platform_result(result, platform, out_jpg)
            out_path = out_jpg
            ext = "jpg"
            if nbytes:
                patch_job(job["id"], {"message": f"Done ({platform} template, {nbytes // (1024 * 1024)}MB)"})
        else:
            result.convert("RGB").save(out_jpg, "JPEG", quality=94, optimize=True)
            out_path = out_jpg
            ext = "jpg"
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
