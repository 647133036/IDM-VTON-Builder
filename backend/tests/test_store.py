import pytest
import store as store_mod
from store import (
    ETHNICITY_MIGRATION,
    add_base,
    add_ethnicity,
    count_queued,
    count_running,
    create_job,
    delete_base,
    delete_ethnicity,
    list_bases,
    list_ethnicities,
    patch_job,
    seed_bases_if_empty,
    stats,
    update_base,
    update_ethnicity,
    update_settings,
)
from store import (
    settings as get_settings,
)


def test_seed_bases_has_five_catalog_entries():
    seed_bases_if_empty()
    bases = list_bases()
    assert {b["id"] for b in bases} >= {
        "base_women_01",
        "base_men_01",
        "base_girl_01",
        "base_boy_01",
        "base_baby_01",
    }


def test_seed_bases_ethnicity_values():
    seed_bases_if_empty()
    by_id = {b["id"]: b for b in list_bases()}
    assert by_id["base_women_01"]["ethnicity"] == "east_asia"
    assert by_id["base_men_01"]["ethnicity"] == "western_europe"
    assert by_id["base_girl_01"]["audience"] == "girls"
    assert by_id["base_baby_01"]["age_band"] == "baby"


def test_add_and_update_base():
    meta = {
        "id": "base_test_01",
        "name": "测试底座",
        "audience": "women",
        "age_band": "adult",
        "ethnicity": "east_asia",
        "pose": "front",
        "file": "test.png",
        "notes": "",
        "active": True,
        "builtin": False,
    }
    add_base(meta)
    updated = update_base("base_test_01", {"ethnicity": "north_america", "name": "改名"})
    assert updated["ethnicity"] == "north_america"
    assert updated["name"] == "改名"


def test_update_base_missing_raises():
    with pytest.raises(KeyError):
        update_base("base_nope", {"name": "x"})


def test_delete_base_removes_from_state():
    add_base({"id": "base_del", "name": "x", "audience": "women", "file": "x.png"})
    delete_base("base_del")
    assert all(b["id"] != "base_del" for b in list_bases())


def test_delete_base_missing_raises():
    with pytest.raises(KeyError):
        delete_base("base_nope")


def test_ethnicity_crud():
    add_ethnicity({"value": "scandinavia", "label": "北欧"})
    assert any(e["value"] == "scandinavia" for e in list_ethnicities())
    update_ethnicity("scandinavia", {"label": "斯堪的纳维亚"})
    entry = next(e for e in list_ethnicities() if e["value"] == "scandinavia")
    assert entry["label"] == "斯堪的纳维亚"
    delete_ethnicity("scandinavia")
    assert all(e["value"] != "scandinavia" for e in list_ethnicities())


def test_add_ethnicity_duplicate_raises():
    add_ethnicity({"value": "dup", "label": "x"})
    with pytest.raises(KeyError):
        add_ethnicity({"value": "dup", "label": "y"})


def test_delete_ethnicity_resets_bases_to_east_asia():
    add_ethnicity({"value": "temp_eth", "label": "临时"})
    add_base({"id": "base_t", "name": "x", "ethnicity": "temp_eth", "file": "x.png"})
    delete_ethnicity("temp_eth")
    base = next(b for b in list_bases() if b["id"] == "base_t")
    assert base["ethnicity"] == "east_asia"


def test_ethnicity_migration_mapping():
    assert ETHNICITY_MIGRATION["slav"] == "western_europe"
    assert ETHNICITY_MIGRATION["east"] == "east_asia"
    assert ETHNICITY_MIGRATION["west"] == "western_europe"
    assert ETHNICITY_MIGRATION["north"] == "north_america"
    assert ETHNICITY_MIGRATION["mongol"] == "east_asia"


def test_list_bases_migrates_unknown_ethnicity():
    add_base({"id": "base_mig", "name": "x", "ethnicity": "slav", "file": "x.png"})
    base = next(b for b in list_bases() if b["id"] == "base_mig")
    assert base["ethnicity"] == "western_europe"


def test_job_lifecycle():
    seed_bases_if_empty()
    job = create_job({"base_id": "base_women_01", "source_name": "a.jpg"})
    assert job["status"] == "queued"
    assert job["audience"] == "auto"

    claimed = store_mod.claim_next_job()
    assert claimed is not None and claimed["id"] == job["id"]
    assert claimed["status"] == "running"

    assert count_running() == 1
    assert count_queued() == 0

    patch_job(job["id"], {"status": "done", "progress": 100})
    assert store_mod.get_job(job["id"])["status"] == "done"


def test_claim_next_job_returns_none_when_empty():
    assert store_mod.claim_next_job() is None


def test_patch_job_missing_raises():
    with pytest.raises(KeyError):
        patch_job("job_nope", {"status": "done"})


def test_settings_and_stats():
    default = get_settings()
    assert "output_long_side" in default
    update_settings({"store_name": "MannequinForge"})
    assert get_settings()["store_name"] == "MannequinForge"

    s = stats()
    assert set(s) >= {"total", "queued", "running", "done", "failed", "bases"}


def test_delete_base_exact_name_only(client=None):
    """P3-5: delete_base 只按精确名删,不误删 base_10.png。"""
    import store as s

    s.seed_bases_if_empty()
    # 先放一个 base_10.png(名字前缀与 base_1 冲突)。
    (s.BASES / "base_10.png").write_bytes(b"x")
    delete_base("base_women_01")
    # base_10.png 必须仍在。
    assert (s.BASES / "base_10.png").exists()


def test_delete_all_bases_does_not_reseed(client=None):
    """P3-6: 删除全部底座后,再次 list_bases 不自动补种。"""
    import store as s

    s.seed_bases_if_empty()
    for b in list_bases():
        delete_base(b["id"])
    assert list_bases() == []
    # 二次确认仍为空。
    assert list_bases() == []
