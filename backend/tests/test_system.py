"""System-level end-to-end test: drive one full try-on job through the
real worker loop (threads claiming jobs) and assert the pipeline lands
as `done` with downloadable artifacts."""

import time
from pathlib import Path

import pytest
from PIL import Image

from store import (
    BASES,
    RESULTS,
    UPLOADS,
    claim_next_job,
    list_jobs,
    seed_bases_if_empty,
)
from worker import WorkerPool


def _mk(uploads_dir, name, color):
    p = uploads_dir / name
    Image.new("RGB", (200, 300), color).save(p, "JPEG")
    return p


def _copy_seed_base(src_name="women_01.png"):
    import shutil

    src = Path(__file__).resolve().parent.parent / "data" / "bases" / src_name
    if src.exists():
        shutil.copy(src, BASES / src_name)


def test_full_worker_loop_drives_job_to_done(client, auth_headers):
    seed_bases_if_empty()
    _copy_seed_base()
    _mk(UPLOADS, "sys_ladydress.jpg", (210, 40, 90))

    r = client.post(
        "/api/jobs/batch",
        data={
            "base_id": "base_women_01",
            "audience": "auto",
            "garment_type": "auto",
            "output_format": "jpg",
            "background": "white",
        },
        files=[("files", ("sys_ladydress.jpg", open(UPLOADS / "sys_ladydress.jpg", "rb").read(), "image/jpeg"))],
        headers=auth_headers,
    )
    assert r.status_code == 200 and r.json()["count"] == 1
    job_id = r.json()["jobs"][0]["id"]

    pool = WorkerPool(workers=1)
    pool.start()
    try:
        deadline = time.time() + 30
        final = None
        while time.time() < deadline:
            time.sleep(0.3)
            row = next((j for j in list_jobs() if j["id"] == job_id), None)
            if row and row["status"] in ("done", "failed"):
                final = row
                break
        assert final is not None, "worker did not finish in 30s"
        assert final["status"] == "done", final.get("error")
        assert final["progress"] == 100
        assert final["result_url"]

        dl = client.get(f"/api/download/{job_id}", headers=auth_headers)
        assert dl.status_code == 200
        assert len(dl.content) > 1000
    finally:
        pool.join()


def test_worker_marks_missing_source_as_failed():
    seed_bases_if_empty()
    _mk(UPLOADS, "sys_badsrc.jpg", (5, 5, 5))
    from store import create_job

    job = create_job({
        "base_id": "base_women_01",
        "audience": "women",
        "garment_type": "upper",
        "source_name": "ghost_not_uploaded.jpg",
        "source_url": None,
    })

    pool = WorkerPool(workers=1)
    pool.start()
    try:
        deadline = time.time() + 20
        final = None
        while time.time() < deadline:
            time.sleep(0.3)
            row = next((j for j in list_jobs() if j["id"] == job["id"]), None)
            if row and row["status"] in ("done", "failed"):
                final = row
                break
        assert final is not None
        assert final["status"] == "failed"
        assert "Garment source image missing" in final.get("error", "")
    finally:
        pool.join()