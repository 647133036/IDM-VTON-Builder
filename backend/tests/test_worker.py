import copy
import os

import pytest
from PIL import Image
from store import (
    claim_next_job,
    create_job,
    get_job,
    patch_job,
    seed_bases_if_empty,
)
from worker import process_job


def _make_png(path, size=(256, 384), color=(30, 120, 200)):
    Image.new("RGB", size, color).save(path)


def _make_sku_png(path, size=(200, 200), color=(200, 40, 40)):
    """白底 + 中央色块的平铺 SKU 图(composite 路径 garment_alpha 需要背景对比)。"""
    import numpy as np

    w, h = size
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[h // 5 : h * 4 // 5, w // 5 : w * 4 // 5] = color
    Image.fromarray(arr, "RGB").save(path)


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
    _make_sku_png(src, (200, 200), (200, 40, 40))
    base_table = [
        {
            "id": "base_women_01",
            "name": "女装A",
            "audience": "women",
            "age_band": "adult",
            "ethnicity": "east_asia",
            "file": None,
            "path": str(ready_base_image),
        }
    ]
    job = create_job(
        {
            "base_id": "base_women_01",
            "audience": "auto",
            "garment_type": "auto",
            "source_name": "ladies_dress.jpg",
            "source_url": "/files/uploads/ladies_dress.jpg",
            "title": "dress",
        }
    )
    process_job(copy.deepcopy(job), base_table)
    final = get_job(job["id"])
    assert final["status"] == "done"
    assert final["result_url"]
    assert final["preview_url"]
    assert get_job(job["id"])["audience"] == "women"
    assert get_job(job["id"])["garment_type"] == "dress"
    # 真实 compose 链路(CPU composite 兜底)的印花保真分/档位/多 seed 元数据落库
    assert final.get("print_fidelity", 0) >= 0.99  # 纯色款 → 满分
    assert final.get("delivery_tier") == "auto"
    assert final.get("multiseed", {}).get("n_seeds", 0) >= 1


def test_process_job_missing_base_file_fails(tmp_path):
    src = tmp_path / "src.jpg"
    _make_png(src)
    job = create_job(
        {
            "base_id": "base_women_01",
            "audience": "auto",
            "garment_type": "upper",
            "source_name": "x.jpg",
            "source_url": None,
        }
    )
    base_table = [
        {
            "id": "base_women_01",
            "file": "does_not_exist.png",
            "path": None,
            "audience": "women",
            "ethnicity": "east_asia",
        }
    ]
    with pytest.raises(RuntimeError):
        process_job(job, base_table)


def test_process_job_missing_source_fails(tmp_path, ready_base_image):
    job = create_job(
        {
            "base_id": "base_women_01",
            "audience": "women",
            "garment_type": "upper",
            "source_name": "nope.jpg",
            "source_url": None,
        }
    )
    base_table = [
        {
            "id": "base_women_01",
            "file": None,
            "path": str(ready_base_image),
            "audience": "women",
            "ethnicity": "east_asia",
        }
    ]
    with pytest.raises(RuntimeError):
        process_job(job, base_table)


def test_workflow_claim_and_fail_marks_job():
    seed_bases_if_empty()
    while claim_next_job() is not None:
        pass
    job = create_job({"base_id": "base_women_01", "source_name": "x.jpg"})
    claimed = claim_next_job()
    assert claimed and claimed["id"] == job["id"]
    patch_job(job["id"], {"status": "failed", "error": "boom", "progress": 100})
    assert get_job(job["id"])["status"] == "failed"


def _stub_compose_factory(fidelity_by_seed=None, fidelity=0.95):
    """返回 compose_tryon 替身: 写入 print_fidelity/delivery_tier 到 img.info。"""

    def _compose(base_path, src, audience, garment_type, long_side, background, ethnicity):
        img = Image.new("RGB", (96, 128), (120, 60, 60))
        if fidelity_by_seed is not None:
            seed = int(os.environ.get("VTON_SEED", "0"))
            pf = fidelity_by_seed.get(seed, 0.5)
        else:
            pf = fidelity
        img.info["print_fidelity"] = pf
        img.info["delivery_tier"] = "auto" if pf >= 0.70 else "review"
        img.info["engine"] = "stub"
        return img

    return _compose


def test_process_job_persists_print_fidelity_tier(tmp_path, ready_base_image, monkeypatch):
    """compose 写入 print_fidelity/delivery_tier 时落库 job 元数据。"""
    import worker as worker_mod

    monkeypatch.setenv("VTON_MULTI_SEED", "1")
    monkeypatch.setattr(worker_mod, "compose_tryon", _stub_compose_factory(fidelity=0.95))

    import store as store_mod

    src = store_mod.UPLOADS / "print_tee.jpg"
    _make_png(src, (200, 200), (200, 40, 40))
    base_table = [
        {
            "id": "base_women_01",
            "file": None,
            "path": str(ready_base_image),
            "audience": "women",
            "ethnicity": "east_asia",
        }
    ]
    job = create_job({"base_id": "base_women_01", "source_name": "print_tee.jpg", "title": "tee"})
    process_job(copy.deepcopy(job), base_table)
    final = get_job(job["id"])
    assert final["status"] == "done"
    assert final["print_fidelity"] == 0.95
    assert final["delivery_tier"] == "auto"


def test_process_job_multiseed_picks_best_print_fidelity(tmp_path, ready_base_image, monkeypatch):
    """多 seed 按印花保真分择优, multiseed 元数据(含 scores)落库。"""
    import worker as worker_mod

    monkeypatch.setenv("VTON_MULTI_SEED", "2")
    # refit_seeds_default() = [42, 1042]: 1042 分高 → 应被选中
    monkeypatch.setattr(
        worker_mod,
        "compose_tryon",
        _stub_compose_factory(fidelity_by_seed={42: 0.40, 1042: 0.90}),
    )

    import store as store_mod

    src = store_mod.UPLOADS / "print_tee2.jpg"
    _make_png(src, (200, 200), (200, 40, 40))
    base_table = [
        {
            "id": "base_women_01",
            "file": None,
            "path": str(ready_base_image),
            "audience": "women",
            "ethnicity": "east_asia",
        }
    ]
    job = create_job({"base_id": "base_women_01", "source_name": "print_tee2.jpg", "title": "tee2"})
    process_job(copy.deepcopy(job), base_table)
    final = get_job(job["id"])
    assert final["status"] == "done"
    assert final["print_fidelity"] == 0.90
    assert final["delivery_tier"] == "auto"
    ms = final.get("multiseed") or {}
    assert ms.get("n_seeds") == 2
    assert ms.get("chosen_seed") == 1042
    assert ms.get("scores", {}).get("42") == 0.40
    assert ms.get("scores", {}).get("1042") == 0.90
