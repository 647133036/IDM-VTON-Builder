import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("STORE_ADMIN_PASSWORD", "admin123")
os.environ.setdefault("STORE_STAFF_PASSWORD", "staff123")
os.environ.setdefault("STORE_JWT_SECRET", "test-secret-key-for-unit-tests-only-32bytes!")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store as store_mod

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

# Web 栈(main/jwt): DSW 的 CPU 子集环境可能缺 jwt 等依赖。import 失败 → 标记不可用,
# web 夹具(client/auth_headers/seeded_client)运行时 skip, 但 CPU 子集
# (test_platform_presets 等)的 collect 不再被 conftest 阻塞(pipeline audit 第 11 条 conftest 拆分)。
try:
    from fastapi.testclient import TestClient
    from main import app

    _WEB_STACK_OK = True
except Exception:  # noqa: BLE001 - 有意宽捕获: web 栈任何 import 失败(缺 jwt/缺驱动等)都降级, 不阻塞 CPU 子集
    TestClient = None
    app = None
    _WEB_STACK_OK = False


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
        # 清理 BASES/TRASH 消除测试间顺序依赖(NEW-L5)。
        for d in (store_mod.UPLOADS, store_mod.RESULTS, store_mod.BASES, store_mod.TRASH):
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
    if not _WEB_STACK_OK:
        pytest.skip("web stack 本环境不可用(main/jwt 缺依赖)")
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
