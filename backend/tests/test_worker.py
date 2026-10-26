import copy

import pytest
from PIL import Image

from store import (
    add_base,
    claim_next_job,
    create_job,
    get_job,
    patch_job,
    seed_bases_if_empty,
    settings as get_settings,
)
from worker import process_job


def _make_png(path, size=(256, 384), color=(30, 120, 200)):
    Image.new("RGB", size, color).save(path)


@pytest.fixture()
def ready_base_image(tmp_path):
    p = tmp_path / "women_01.png"
    _make_png(p)
    return p


@pytest.fixture(autouse=True)
def point_bases_dir(tmp_path, monkeypatch):
    import store as store_mod

    monkeypatch.setattr(store_mod, "BASES", tmp_path)
    return store_mod.BASES


def _mk_table():
    seed_bases_if_empty()
    return [
        {
            "id": "base_women_01",
            "name": "女装A",
            "audience": "women",
            "age_band": "adult",
            "ethnicity": "east_asia",
            "file": "women_01.png",
            "path": None,
        }
    ]


def test_process_job_done_with_base_ethnicity(tmp_path, ready_base_image):
    import store as store_mod

    src = store_mod.UPLOADS / "ladies_dress.jpg"
    _make_png(src, (200, 200), (200, 40, 40))
    base_table = [{
        "id": "base_women_01",
        "name": "女装A",
        "audience": "women",
        "age_band": "adult",
        "ethnicity": "east_asia",
        "file": None,
        "path": str(ready_base_image),
    }]
    job = create_job({
        "base_id": "base_women_01",
        "audience": "auto",
        "garment_type": "auto",
        "source_name": "ladies_dress.jpg",
        "source_url": "/files/uploads/ladies_dress.jpg",
        "title": "dress",
    })
    process_job(copy.deepcopy(job), base_table)
    final = get_job(job["id"])
    assert final["status"] == "done"
    assert final["result_url"]
    assert final["preview_url"]
    assert get_job(job["id"])["audience"] == "women"
    assert get_job(job["id"])["garment_type"] == "dress"


def test_process_job_missing_base_file_fails(tmp_path):
    src = tmp_path / "src.jpg"
    _make_png(src)
    job = create_job({
        "base_id": "base_women_01",
        "audience": "auto",
        "garment_type": "upper",
        "source_name": "x.jpg",
        "source_url": None,
    })
    base_table = [{"id": "base_women_01", "file": "does_not_exist.png", "path": None, "audience": "women", "ethnicity": "east_asia"}]
    with pytest.raises(RuntimeError):
        process_job(job, base_table)


def test_process_job_missing_source_fails(tmp_path, ready_base_image):
    job = create_job({
        "base_id": "base_women_01",
        "audience": "women",
        "garment_type": "upper",
        "source_name": "nope.jpg",
        "source_url": None,
    })
    base_table = [{"id": "base_women_01", "file": None, "path": str(ready_base_image), "audience": "women", "ethnicity": "east_asia"}]
    with pytest.raises(RuntimeError):
        process_job(job, base_table)


def test_workflow_claim_and_fail_marks_job():
    seed_bases_if_empty()
    job = create_job({"base_id": "base_women_01", "source_name": "x.jpg"})
    claimed = claim_next_job()
    assert claimed and claimed["id"] == job["id"]
    patch_job(job["id"], {"status": "failed", "error": "boom", "progress": 100})
    assert get_job(job["id"])["status"] == "failed"