import contextlib
import copy
import fcntl
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
JOBS_DIR = DATA / "jobs"
UPLOADS = DATA / "uploads"
RESULTS = DATA / "results"
BASES = DATA / "bases"
TRASH = DATA / "trash"
STATE_FILE = DATA / "state.json"

for p in (JOBS_DIR, UPLOADS, RESULTS, BASES, TRASH):
    p.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()

ETHNICITY_DEFAULTS = [
    {"value": "western_europe", "label": "Western Europe"},
    {"value": "eastern_europe", "label": "Eastern Europe"},
    {"value": "west_asia", "label": "West Asia"},
    {"value": "central_asia", "label": "Central Asia"},
    {"value": "east_asia", "label": "East Asia"},
    {"value": "southeast_asia", "label": "Southeast Asia"},
    {"value": "africa", "label": "Africa"},
    {"value": "north_america", "label": "North America"},
    {"value": "south_america", "label": "South America"},
    {"value": "oceania", "label": "Oceania"},
]

DEFAULT_STATE = {
    "bases": [],
    "jobs": [],
    "uploads": [],
    "stores": [],
    "ethnicities": ETHNICITY_DEFAULTS,
    # 仅首次启动种子一次;管理员删除全部底座后不再自动补种(P3-6)。
    "_bases_seeded": False,
    "settings": {
        "output_long_side": 2048,
        "background": "white",
        "engine": "idm-vton-composite",
        "store_name": "Atelier Studio",
        "webhook_url": "",
    },
}


def _now() -> int:
    return int(time.time())


def _load_state_locked() -> dict:
    if not STATE_FILE.exists():
        save_state(DEFAULT_STATE)
        return json.loads(json.dumps(DEFAULT_STATE))
    with STATE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in DEFAULT_STATE.items():
        data.setdefault(key, value)
    return data


def load_state() -> dict:
    _lock.acquire()
    try:
        with STATE_FILE.with_suffix(".lock").open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            data = _load_state_locked()
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        return data
    finally:
        _lock.release()


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    tmp.replace(STATE_FILE)
    with contextlib.suppress(OSError):
        os.chmod(STATE_FILE, 0o600)


def with_state(fn):
    _lock.acquire()
    try:
        with STATE_FILE.with_suffix(".lock").open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            state = _load_state_locked()
            result = fn(state)
            save_state(state)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        return result
    finally:
        _lock.release()


def uid(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def embed_url(origin: str, store_id: str = "") -> str:
    store_id = Path(store_id or "").name
    return f"{origin}/embed.html?store={store_id}" if store_id else f"{origin}/embed.html"


def public_url(path: Path) -> str:
    rel = path.relative_to(DATA).as_posix()
    return f"/files/{rel}"


def find_in_dir(directory: Path, name: str) -> Path | None:
    for p in directory.rglob(Path(name).name):
        if p.is_file():
            return p
    return None


def seed_bases_if_empty() -> None:
    def _seed(state: dict):
        # 已种子过就不再补种,管理员删空后保持为空(P3-6)。
        if state.get("_bases_seeded"):
            return
        state["_bases_seeded"] = True
        if not state["bases"]:
            catalog = [
                {
                    "id": "base_women_01",
                    "name": "Women · Front-view Model A",
                    "audience": "women",
                    "age_band": "adult",
                    "ethnicity": "east_asia",
                    "pose": "front",
                    "file": "women_01.png",
                    "notes": "Front standing pose, arms slightly spread; suits tops/dresses",
                },
                {
                    "id": "base_men_01",
                    "name": "Men · Front-view Model A",
                    "audience": "men",
                    "age_band": "adult",
                    "ethnicity": "western_europe",
                    "pose": "front",
                    "file": "men_01.png",
                    "notes": "Front standing pose; suits t-shirts/shirts/knitwear",
                },
                {
                    "id": "base_girl_01",
                    "name": "Girls · Front-view Model",
                    "audience": "girls",
                    "age_band": "kids",
                    "ethnicity": "east_asia",
                    "pose": "front",
                    "file": "girl_01.png",
                    "notes": "Girl about 7 years old; suits kids tops/skirts",
                },
                {
                    "id": "base_boy_01",
                    "name": "Boys · Front-view Model",
                    "audience": "boys",
                    "age_band": "kids",
                    "ethnicity": "east_asia",
                    "pose": "front",
                    "file": "boy_01.png",
                    "notes": "Boy about 7 years old; suits kids t-shirts/shorts",
                },
                {
                    "id": "base_baby_01",
                    "name": "Baby · Front-view Model",
                    "audience": "baby",
                    "age_band": "baby",
                    "ethnicity": "east_asia",
                    "pose": "front",
                    "file": "baby_01.png",
                    "notes": "Baby about 2 years old; suits onesies/rompers",
                },
            ]
            now = _now()
            for item in catalog:
                src = BASES / item["file"]
                state["bases"].append(
                    {
                        **item,
                        "url": public_url(src) if src.exists() else "",
                        "active": True,
                        "builtin": True,
                        "created_at": now,
                    }
                )
            return
        fallback = {
            "base_women_01": "east_asia",
            "base_men_01": "western_europe",
            "base_girl_01": "east_asia",
            "base_boy_01": "east_asia",
            "base_baby_01": "east_asia",
        }
        for item in state["bases"]:
            if "ethnicity" not in item:
                item["ethnicity"] = fallback.get(item["id"], "east_asia")

    with_state(_seed)


ETHNICITY_MIGRATION = {
    "slav": "western_europe",
    "east": "east_asia",
    "west": "western_europe",
    "north": "north_america",
    "mongol": "east_asia",
}


def list_bases() -> list[dict]:
    seed_bases_if_empty()
    valid = {e["value"] for e in list_ethnicities()}
    data = load_state()
    for b in data["bases"]:
        eth = b.get("ethnicity")
        if eth and eth not in valid:
            b["ethnicity"] = ETHNICITY_MIGRATION.get(eth, "east_asia")
    return data["bases"]


def add_base(meta: dict) -> dict:
    def _add(state: dict):
        state["bases"].append(meta)
        return meta

    return with_state(_add)


def update_base(base_id: str, patch: dict) -> dict:
    def _upd(state: dict):
        for item in state["bases"]:
            if item["id"] == base_id:
                item.update(patch)
                return item
        raise KeyError(base_id)

    return with_state(_upd)


def delete_base(base_id: str) -> None:
    meta_file = None

    def _del(state: dict):
        nonlocal meta_file
        for b in state["bases"]:
            if b["id"] == base_id:
                meta_file = b.get("file")
                break
        else:
            raise KeyError(base_id)
        state["bases"] = [b for b in state["bases"] if b["id"] != base_id]

    with_state(_del)

    def _trash(p: Path) -> None:
        dest = TRASH / p.name
        i = 1
        while dest.exists():
            dest = TRASH / f"{p.stem}_{i}{p.suffix}"
            i += 1
        p.replace(dest)

    # 优先按元数据 file 删除内置底座(women_01.png 而非 base_women_01.png, NEW-L2);
    # 无元数据时回退到 <base_id>.* 精确名匹配。
    if meta_file:
        target = BASES / Path(meta_file).name
        if target.is_file():
            _trash(target)
            return
    for p in sorted(BASES.iterdir()):
        if not p.is_file():
            continue
        if p.stem != base_id:
            continue
        _trash(p)
        break


def create_job(payload: dict) -> dict:
    def _add(state: dict):
        job = {
            "id": uid("job"),
            "status": "queued",
            "created_at": _now(),
            "updated_at": _now(),
            "base_id": payload["base_id"],
            "audience": payload.get("audience", "auto"),
            "garment_type": payload.get("garment_type", "upper"),
            "output_format": payload.get("output_format", "jpg"),
            "background": payload.get("background", state["settings"]["background"]),
            "sku": payload.get("sku", ""),
            "title": payload.get("title", ""),
            "prompt": payload.get("prompt", ""),
            "source_name": payload.get("source_name", ""),
            "source_url": payload.get("source_url", ""),
            "bound_store_id": payload.get("bound_store_id", ""),
            "result_url": "",
            "preview_url": "",
            "progress": 0,
            "message": "Queued",
            "error": "",
        }
        state["jobs"].insert(0, job)
        return job

    return with_state(_add)


def patch_job(job_id: str, patch: dict) -> dict:
    def _upd(state: dict):
        for item in state["jobs"]:
            if item["id"] == job_id:
                item.update(patch)
                item["updated_at"] = _now()
                return item
        raise KeyError(job_id)

    return with_state(_upd)


def get_job(job_id: str) -> dict:
    for item in load_state()["jobs"]:
        if item["id"] == job_id:
            return item
    raise KeyError(job_id)


def list_jobs() -> list[dict]:
    return load_state()["jobs"]


def claim_next_job() -> dict | None:
    def _claim(state: dict):
        for item in state["jobs"]:
            if item["status"] == "queued":
                item["status"] = "running"
                item["progress"] = 5
                item["message"] = "Worker claimed, processing started"
                item["updated_at"] = _now()
                return copy.deepcopy(item)
        return None

    return with_state(_claim)


def count_queued() -> int:
    return sum(1 for j in load_state()["jobs"] if j["status"] == "queued")


def count_running() -> int:
    return sum(1 for j in load_state()["jobs"] if j["status"] == "running")


def settings() -> dict:
    return load_state()["settings"]


def update_settings(patch: dict) -> dict:
    def _upd(state: dict):
        state["settings"].update(patch)
        return state["settings"]

    return with_state(_upd)


def list_ethnicities() -> list[dict]:
    return load_state()["ethnicities"]


def add_ethnicity(entry: dict) -> dict:
    def _add(state: dict):
        for item in state["ethnicities"]:
            if item["value"] == entry["value"]:
                raise KeyError("duplicate")
        state["ethnicities"].append(entry)
        return entry

    return with_state(_add)


def update_ethnicity(value: str, patch: dict) -> dict:
    def _upd(state: dict):
        if value not in {e["value"] for e in state["ethnicities"]}:
            raise KeyError(value)
        for e in state["ethnicities"]:
            if e["value"] == value:
                e.update(patch)
                return e
        raise KeyError(value)

    return with_state(_upd)


def delete_ethnicity(value: str) -> None:
    def _del(state: dict):
        remaining = [e for e in state["ethnicities"] if e["value"] != value]
        if len(remaining) == len(state["ethnicities"]):
            raise KeyError(value)
        state["ethnicities"] = remaining
        # 被删的 ethnic 从底座上摘掉;回退目标必须仍然存在(NEW-L3)。
        fallback = (
            "east_asia"
            if any(e["value"] == "east_asia" for e in remaining)
            else (remaining[0]["value"] if remaining else "")
        )
        for b in state["bases"]:
            if b.get("ethnicity") == value:
                b["ethnicity"] = fallback or ""

    with_state(_del)


def list_stores() -> list[dict]:
    return load_state()["stores"]


def get_store(store_id: str) -> dict | None:
    return next((s for s in load_state()["stores"] if s["id"] == store_id), None)


def add_store(meta: dict) -> dict:
    def _add(state: dict):
        state["stores"].append(meta)
        return meta

    return with_state(_add)


def update_store(store_id: str, patch: dict) -> dict:
    def _upd(state: dict):
        for item in state["stores"]:
            if item["id"] == store_id:
                item.update(patch)
                return item
        raise KeyError(store_id)

    return with_state(_upd)


def delete_store(store_id: str) -> None:
    def _del(state: dict):
        remaining = [s for s in state["stores"] if s["id"] != store_id]
        if len(remaining) == len(state["stores"]):
            raise KeyError(store_id)
        state["stores"] = remaining

    with_state(_del)


def stats() -> dict[str, Any]:
    jobs = list_jobs()
    return {
        "total": len(jobs),
        "queued": sum(1 for j in jobs if j["status"] == "queued"),
        "running": sum(1 for j in jobs if j["status"] == "running"),
        "done": sum(1 for j in jobs if j["status"] == "done"),
        "failed": sum(1 for j in jobs if j["status"] == "failed"),
        "bases": len(list_bases()),
    }
