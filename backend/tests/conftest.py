import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store as store_mod  # noqa: E402

_TEST_DIR = Path(tempfile.mkdtemp(prefix="vton_test_"))

store_mod.DATA = _TEST_DIR
store_mod.JOBS_DIR = _TEST_DIR / "jobs"
store_mod.UPLOADS = _TEST_DIR / "uploads"
store_mod.RESULTS = _TEST_DIR / "results"
store_mod.BASES = _TEST_DIR / "bases"
store_mod.TRASH = _TEST_DIR / "trash"
store_mod.STATE_FILE = _TEST_DIR / "state.json"

for p in (store_mod.JOBS_DIR, store_mod.UPLOADS, store_mod.RESULTS, store_mod.BASES, store_mod.TRASH):
    p.mkdir(parents=True, exist_ok=True)

import worker as worker_mod  # noqa: E402

worker_mod.start_worker = lambda *a, **k: None

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

import main as main_mod  # noqa: E402


def _reset_filesystem():
    state = {
        "bases": [],
        "jobs": [],
        "uploads": [],
        "stores": [],
        "ethnicities": list(store_mod.ETHNICITY_DEFAULTS),
        "settings": {
            "output_long_side": 512,
            "background": "white",
            "engine": "idm-vton-composite",
            "store_name": "Atelier Studio",
            "webhook_url": "",
        },
    }
    store_mod._lock.acquire()
    try:
        tmp = store_mod.STATE_FILE.with_suffix(".test.tmp")
        tmp.write_text(__import__("json").dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(store_mod.STATE_FILE)
        for d in (store_mod.UPLOADS, store_mod.RESULTS):
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
    finally:
        store_mod._lock.release()
    store_mod.seed_bases_if_empty()


@pytest.fixture(autouse=True)
def clean_state():
    _reset_filesystem()
    yield


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture()
def auth_headers(client):
    r = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _make_image(path: Path, size=(320, 480), color=(200, 60, 60)):
    from PIL import Image

    img = Image.new("RGB", size, color)
    img.save(path)
    return path


@pytest.fixture()
def base_image(tmp_path):
    return _make_image(tmp_path / "base.png")


@pytest.fixture()
def garment_image(tmp_path):
    return _make_image(tmp_path / "garment.jpg")


@pytest.fixture(scope="session")
def seeded_client(client):
    client.get("/api/health")
    return client