"""CSV + 图片 ZIP 批量导出。

把一批已完成任务的成品图连同 CSV 清单打包成 ZIP,供运营人员离线修图后
连图带 CSV 回传批量推送(店铺详情页 batch-push 入口)。
"""

from __future__ import annotations

import csv
import io
import zipfile

from PIL import Image
from store import RESULTS, get_job, list_stores

CSV_COLUMNS = [
    "job_id",
    "image_filename",
    "source_name",
    "store_name",
    "store_id",
    "product_id",
    "sku",
    "status",
]


def export_jobs_csv(job_ids: list[str]) -> bytes:
    """生成 ZIP(bytes):CSV 清单 + 所有已完成任务的成品图。

    - job 不存在或未 done 时跳过
    - 图片按 ``{job_id}.{ext}`` 命名,与 CSV 的 image_filename 列一一对应
    - store_name / store_id 从 job.bound_store_id 查店铺,未绑定时留空
    """
    stores = {s["id"]: s for s in list_stores()}
    rows: list[dict] = []
    images: list[tuple[str, bytes]] = []

    for jid in job_ids:
        try:
            job = get_job(jid)
        except KeyError:
            continue
        if job.get("status") != "done":
            continue
        result_url = job.get("result_url", "")
        if not result_url:
            continue
        filename = result_url.split("/")[-1]
        img_path = RESULTS / filename
        if not img_path.is_file():
            continue

        # 统一导出为 JPEG(体积更小,离线修图工具兼容性好)
        ext = "jpg"
        export_name = f"{jid}.{ext}"
        try:
            img = Image.open(img_path)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=95)
            images.append((export_name, buf.getvalue()))
        except (OSError, ValueError):
            # 图损坏时仍保留 CSV 行,图片文件缺失导入时会标注
            export_name = filename
            images.append((export_name, img_path.read_bytes()))

        store_id = job.get("bound_store_id", "")
        store = stores.get(store_id, {})
        rows.append(
            {
                "job_id": jid,
                "image_filename": export_name,
                "source_name": job.get("source_name", ""),
                "store_name": store.get("name", ""),
                "store_id": store_id,
                "product_id": store.get("product_id", ""),
                "sku": job.get("sku", "") or store.get("default_sku", ""),
                "status": job.get("status", ""),
            }
        )

    if not rows:
        return b""

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.csv", csv_buf.getvalue())
        for name, data in images:
            zf.writestr(name, data)

    return zip_buf.getvalue()
