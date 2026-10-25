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

DEFAULT_STATE = {
    "bases": [],
    "jobs": [],
    "uploads": [],
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


def load_state() -> dict:
    if not STATE_FILE.exists():
        save_state(DEFAULT_STATE)
        return json.loads(json.dumps(DEFAULT_STATE))
    with STATE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in DEFAULT_STATE.items():
        data.setdefault(key, value)
    return data


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(STATE_FILE)


def with_state(fn):
    with _lock:
        state = load_state()
        result = fn(state)
        save_state(state)
        return result


def uid(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def public_url(path: Path) -> str:
    rel = path.relative_to(DATA).as_posix()
    return f"/files/{rel}"


def seed_bases_if_empty() -> None:
    def _seed(state: dict):
        if state["bases"]:
            return
        catalog = [
            {
                "id": "base_women_01",
                "name": "女装 · 前视模特 A",
                "audience": "women",
                "age_band": "adult",
                "pose": "front",
                "file": "women_01.png",
                "notes": "正面站姿，手臂微张，适合上衣/连衣裙",
            },
            {
                "id": "base_men_01",
                "name": "男装 · 前视模特 A",
                "audience": "men",
                "age_band": "adult",
                "pose": "front",
                "file": "men_01.png",
                "notes": "正面站姿，适合 T 恤/衬衫/针织",
            },
            {
                "id": "base_girl_01",
                "name": "女童 · 前视模特",
                "audience": "girls",
                "age_band": "kids",
                "pose": "front",
                "file": "girl_01.png",
                "notes": "约 7 岁女童，适合童装上衣/裙子",
            },
            {
                "id": "base_boy_01",
                "name": "男童 · 前视模特",
                "audience": "boys",
                "age_band": "kids",
                "pose": "front",
                "file": "boy_01.png",
                "notes": "约 7 岁男童，适合童装 T 恤/短裤",
            },
            {
                "id": "base_baby_01",
                "name": "婴幼儿 · 前视模特",
                "audience": "baby",
                "age_band": "baby",
                "pose": "front",
                "file": "baby_01.png",
                "notes": "约 2 岁婴幼儿，适合连体衣/哈衣",
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

    with_state(_seed)


def list_bases() -> list[dict]:
    seed_bases_if_empty()
    return load_state()["bases"]


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
    def _del(state: dict):
        remaining = [b for b in state["bases"] if b["id"] != base_id]
        if len(remaining) == len(state["bases"]):
            raise KeyError(base_id)
        state["bases"] = remaining

    with_state(_del)
    src = BASES / f"{base_id}.png"
    if not src.exists():
        src = next((p for p in BASES.iterdir() if p.name.startswith(base_id)), None)
    if src and src.exists():
        dest = TRASH / src.name
        i = 1
        while dest.exists():
            dest = TRASH / f"{src.stem}_{i}{src.suffix}"
            i += 1
        src.replace(dest)


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
            "result_url": "",
            "preview_url": "",
            "progress": 0,
            "message": "已进入队列",
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


def settings() -> dict:
    return load_state()["settings"]


def update_settings(patch: dict) -> dict:
    def _upd(state: dict):
        state["settings"].update(patch)
        return state["settings"]

    return with_state(_upd)


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
