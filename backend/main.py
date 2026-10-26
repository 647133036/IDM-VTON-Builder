from pathlib import Path
from typing import Optional
import shutil
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from auth import USERS, current_user, issue_token
from engine import engine_status, guess_audience, guess_garment_type
from store import (
    BASES,
    DATA,
    RESULTS,
    TRASH,
    UPLOADS,
    add_base,
    add_ethnicity,
    add_store,
    create_job,
    delete_base,
    delete_ethnicity,
    delete_store,
    embed_url,
    find_in_dir,
    get_job,
    get_store,
    list_bases,
    list_ethnicities,
    list_jobs,
    list_stores,
    patch_job,
    public_url,
    seed_bases_if_empty,
    settings as load_settings,
    stats,
    uid,
    update_base,
    update_ethnicity,
    update_settings,
    update_store,
)
from push import push_store
from worker import start_worker

app = FastAPI(title="Atelier Try-On", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

seed_bases_if_empty()
start_worker()


@app.get("/files/{path:path}")
def serve_file(path: str):
    rel = Path(path)
    target = (DATA / rel).resolve()
    if DATA.resolve() not in target.parents and target != DATA.resolve():
        raise HTTPException(status_code=404, detail="文件不存在")
    if target.is_file():
        return FileResponse(target)
    uploads_candidate = DATA / "uploads" / rel.name
    if str(rel).startswith("uploads"):
        found = find_in_dir(UPLOADS, rel.name)
        if found:
            return FileResponse(found)
    if uploads_candidate.is_file():
        return FileResponse(uploads_candidate)
    raise HTTPException(status_code=404, detail="文件不存在")


class LoginBody(BaseModel):
    username: str
    password: str


class SettingsBody(BaseModel):
    output_long_side: Optional[int] = None
    background: Optional[str] = None
    engine: Optional[str] = None
    store_name: Optional[str] = None
    webhook_url: Optional[str] = None


class BasePatch(BaseModel):
    name: Optional[str] = None
    audience: Optional[str] = None
    age_band: Optional[str] = None
    ethnicity: Optional[str] = None
    pose: Optional[str] = None
    notes: Optional[str] = None


class BaseBatchPatch(BaseModel):
    ids: list[str]
    patch: dict[str, Optional[str]] = {}


class EthnicityBody(BaseModel):
    value: str
    label: str


class EthnicityPatch(BaseModel):
    label: Optional[str] = None


class CategoryBody(BaseModel):
    name: str


class StoreBody(BaseModel):
    name: str
    platform: Optional[str] = "other"
    shop_url: Optional[str] = ""
    webhook_url: Optional[str] = ""
    api_key: Optional[str] = ""
    product_id: Optional[str] = ""


class StorePatch(BaseModel):
    name: Optional[str] = None
    platform: Optional[str] = None
    shop_url: Optional[str] = None
    webhook_url: Optional[str] = None
    api_key: Optional[str] = None
    product_id: Optional[str] = None


class MoveBody(BaseModel):
    category: str = "default"
    active: Optional[bool] = None


class PushBody(BaseModel):
    sku: Optional[str] = ""
    product_url: Optional[str] = ""
    product_id: Optional[str] = ""
    store_ids: Optional[list[str]] = []


@app.post("/api/auth/login")
def login(body: LoginBody):
    user = USERS.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    token = issue_token(body.username)
    return {"token": token, "name": user["name"], "role": user["role"], "username": body.username}


@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return user


@app.get("/api/stats")
def api_stats(_: dict = Depends(current_user)):
    return stats()


@app.get("/api/settings")
def api_settings(_: dict = Depends(current_user)):
    return load_settings()


@app.patch("/api/settings")
def api_update_settings(body: SettingsBody, _: dict = Depends(current_user)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    return update_settings(patch)


@app.get("/api/bases")
def api_bases(_: dict = Depends(current_user)):
    seed_bases_if_empty()
    items = list_bases()
    for item in items:
        path = BASES / item.get("file", "")
        if path.exists():
            item["url"] = public_url(path)
    return items


@app.post("/api/bases")
async def api_add_base(
    name: str = Form(...),
    audience: str = Form("women"),
    age_band: str = Form("adult"),
    ethnicity: str = Form("east_asia"),
    notes: str = Form(""),
    file: UploadFile = File(...),
    _: dict = Depends(current_user),
):
    if ethnicity not in {e["value"] for e in list_ethnicities()}:
        raise HTTPException(status_code=400, detail=f"未知人种标签: {ethnicity}")
    ext = Path(file.filename or "base.png").suffix.lower() or ".png"
    base_id = uid("base")
    filename = f"{base_id}{ext}"
    dest = BASES / filename
    dest.write_bytes(await file.read())
    meta = {
        "id": base_id,
        "name": name,
        "audience": audience,
        "age_band": age_band,
        "ethnicity": ethnicity,
        "pose": "front",
        "file": filename,
        "notes": notes,
        "url": public_url(dest),
        "active": True,
        "builtin": False,
        "created_at": int(dest.stat().st_mtime),
    }
    return add_base(meta)


@app.patch("/api/bases/{base_id}")
def api_patch_base(base_id: str, body: BasePatch, _: dict = Depends(current_user)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    if "ethnicity" in patch and patch["ethnicity"] not in {e["value"] for e in list_ethnicities()}:
        raise HTTPException(status_code=400, detail=f"未知人种标签: {patch['ethnicity']}")
    try:
        return update_base(base_id, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="底座不存在") from exc


@app.post("/api/bases/batch/patch")
def api_patch_bases_batch(body: BaseBatchPatch, _: dict = Depends(current_user)):
    patch = {k: v for k, v in body.patch.items() if v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="无可更新的属性")
    allowed = {"name", "audience", "age_band", "ethnicity", "pose", "notes"}
    patch = {k: v for k, v in patch.items() if k in allowed}
    if "ethnicity" in patch and patch["ethnicity"] not in {e["value"] for e in list_ethnicities()}:
        raise HTTPException(status_code=400, detail=f"未知人种标签: {patch['ethnicity']}")
    updated = []
    for base_id in body.ids:
        try:
            updated.append(update_base(base_id, dict(patch)))
        except KeyError:
            continue
    return {"ok": True, "count": len(updated), "bases": updated}


@app.delete("/api/bases/{base_id}")
def api_delete_base(base_id: str, _: dict = Depends(current_user)):
    try:
        delete_base(base_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="底座不存在") from exc
    return {"ok": True, "id": base_id}


@app.get("/api/ethnicities")
def api_ethnicities(_: dict = Depends(current_user)):
    return list_ethnicities()


@app.post("/api/ethnicities")
def api_add_ethnicity(body: EthnicityBody, _: dict = Depends(current_user)):
    value = body.value.strip()
    label = body.label.strip()
    if not value or not label:
        raise HTTPException(status_code=400, detail="人种 value 与 label 不能为空")
    try:
        return add_ethnicity({"value": value, "label": label})
    except KeyError:
        raise HTTPException(status_code=400, detail="人种标签已存在")


@app.patch("/api/ethnicities/{value}")
def api_patch_ethnicity(value: str, body: EthnicityPatch, _: dict = Depends(current_user)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return update_ethnicity(value, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="人种标签不存在") from exc


@app.delete("/api/ethnicities/{value}")
def api_delete_ethnicity(value: str, _: dict = Depends(current_user)):
    try:
        delete_ethnicity(value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="人种标签不存在") from exc
    return {"ok": True, "value": value}


def _origin(request: Request) -> str:
    forwarded_host = request.headers.get("x-forwarded-host")
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_port = request.headers.get("x-forwarded-port")
    if forwarded_host:
        proto = forwarded_proto or "http"
        host = forwarded_host
        if forwarded_port and forwarded_port not in ("80", "443"):
            host = f"{host}:{forwarded_port}"
        return f"{proto}://{host}"
    base = str(request.base_url).rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    return base


@app.get("/api/stores")
def api_stores(request: Request, _: dict = Depends(current_user)):
    origin = _origin(request)
    items = []
    for s in list_stores():
        item = dict(s)
        item["embed_url"] = embed_url(origin, s["id"])
        items.append(item)
    return items


@app.post("/api/stores")
def api_add_store(body: StoreBody, request: Request, _: dict = Depends(current_user)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="店铺名称不能为空")
    meta = {
        "id": uid("store"),
        "name": name,
        "platform": body.platform or "other",
        "shop_url": body.shop_url or "",
        "webhook_url": body.webhook_url or "",
        "api_key": body.api_key or "",
        "product_id": body.product_id or "",
        "created_at": int(__import__("time").time()),
        "active": True,
    }
    created = add_store(meta)
    created["embed_url"] = embed_url(_origin(request), created["id"])
    return created


@app.patch("/api/stores/{store_id}")
def api_patch_store(store_id: str, body: StorePatch, _: dict = Depends(current_user)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return update_store(store_id, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="店铺不存在") from exc


@app.delete("/api/stores/{store_id}")
def api_delete_store(store_id: str, _: dict = Depends(current_user)):
    try:
        delete_store(store_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="店铺不存在") from exc
    return {"ok": True, "id": store_id}


@app.get("/api/jobs")
def api_jobs(_: dict = Depends(current_user)):
    return list_jobs()


@app.get("/api/uploads")
def api_list_uploads(category: Optional[str] = None, _: dict = Depends(current_user)):
    items = []
    if category:
        cat_dir = UPLOADS / Path(category).name
        if cat_dir.exists():
            for p in sorted(cat_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    items.append({
                        "name": p.name,
                        "url": public_url(p),
                        "size": p.stat().st_size,
                        "mtime": int(p.stat().st_mtime),
                        "category": Path(category).name,
                    })
    else:
        for p in sorted(UPLOADS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                items.append({
                    "name": p.name,
                    "url": public_url(p),
                    "size": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                    "category": "default",
                })
            elif p.is_dir():
                for f in sorted(p.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                        items.append({
                            "name": f.name,
                            "url": public_url(f),
                            "size": f.stat().st_size,
                            "mtime": int(f.stat().st_mtime),
                            "category": p.name,
                        })
    return items


@app.delete("/api/uploads/{filename}")
def api_delete_upload(filename: str, _: dict = Depends(current_user)):
    safe_name = Path(filename).name
    p = UPLOADS / safe_name
    if not p.exists():
        for d in UPLOADS.iterdir():
            if d.is_dir():
                candidate = d / safe_name
                if candidate.exists():
                    p = candidate
                    break
    if not p.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    trash_dest = TRASH / f"upload_{p.parent.name}_{safe_name}"
    if trash_dest.exists():
        trash_dest.unlink()
    p.replace(trash_dest)
    return {"ok": True, "filename": safe_name}


class MoveBody(BaseModel):
    category: str = "default"


@app.post("/api/uploads/{filename}/move")
def api_move_upload(filename: str, body: MoveBody, _: dict = Depends(current_user)):
    safe_name = Path(filename).name
    p = UPLOADS / safe_name
    if not p.exists():
        for d in UPLOADS.iterdir():
            if d.is_dir():
                candidate = d / safe_name
                if candidate.exists():
                    p = candidate
                    break
    if not p.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    category = Path(body.category).name
    if p.parent.name == category:
        return {"ok": True, "filename": safe_name, "category": category}
    cat_dir = UPLOADS / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    dest = cat_dir / safe_name
    if dest.exists():
        dest.unlink()
    p.replace(dest)
    return {"ok": True, "filename": safe_name, "category": category}


@app.post("/api/uploads/categories")
def api_add_upload_category(body: CategoryBody, _: dict = Depends(current_user)):
    name = Path(body.name).name.strip()
    if not name or name in ("default", "."):
        raise HTTPException(status_code=400, detail="分类名称无效")
    cat_dir = UPLOADS / name
    cat_dir.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "category": name}


@app.patch("/api/uploads/categories/{category}")
def api_rename_upload_category(category: str, body: CategoryBody, _: dict = Depends(current_user)):
    src_dir = UPLOADS / Path(category).name
    new_name = Path(body.name).name.strip()
    if not src_dir.is_dir():
        raise HTTPException(status_code=404, detail="分类不存在")
    if not new_name or new_name in ("default", "."):
        raise HTTPException(status_code=400, detail="分类名称无效")
    if new_name == src_dir.name:
        return {"ok": True, "category": new_name}
    if (UPLOADS / new_name).exists():
        raise HTTPException(status_code=400, detail=f"分类已存在: {new_name}")
    src_dir.rename(UPLOADS / new_name)
    return {"ok": True, "category": new_name}


@app.delete("/api/uploads/categories/{category}")
def api_delete_upload_category(category: str, _: dict = Depends(current_user)):
    cat_dir = UPLOADS / Path(category).name
    if not cat_dir.is_dir():
        raise HTTPException(status_code=404, detail="分类不存在")
    moved = 0
    for p in cat_dir.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            dest = UPLOADS / p.name
            i = 1
            while dest.exists():
                dest = UPLOADS / f"{p.stem}_{i}{p.suffix}"
                i += 1
            p.replace(dest)
            moved += 1
    try:
        cat_dir.rmdir()
    except OSError:
        cat_dir.rename(TRASH / f"category_{cat_dir.name}")
    return {"ok": True, "category": category, "moved_to_default": moved}


@app.get("/api/uploads/categories")
def api_upload_categories(_: dict = Depends(current_user)):
    categories = {"default": 0}
    for p in UPLOADS.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            categories["default"] += 1
        elif p.is_dir():
            count = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
            categories[p.name] = count
    return categories


@app.get("/api/uploads/{category}/{filename}")
def api_get_upload(category: str, filename: str, _: dict = Depends(current_user)):
    safe_category = Path(category).name
    safe_name = Path(filename).name
    p = UPLOADS / safe_category / safe_name
    if not p.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return {
        "name": p.name,
        "url": public_url(p),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
        "category": safe_category,
    }


@app.post("/api/uploads")
async def api_upload_to_library(
    file: UploadFile = File(...),
    category: str = Form(""),
    _: dict = Depends(current_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    safe_name = f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    cat = Path(category).name if category else "default"
    cat_dir = UPLOADS / cat
    cat_dir.mkdir(parents=True, exist_ok=True)
    dest = cat_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return {
        "name": safe_name,
        "url": public_url(dest),
        "size": dest.stat().st_size,
        "category": cat,
    }


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, _: dict = Depends(current_user)):
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc


@app.post("/api/jobs/batch")
async def api_batch(
    files: list[UploadFile] = File(...),
    base_id: str = Form(...),
    audience: str = Form("auto"),
    garment_type: str = Form("auto"),
    output_format: str = Form("jpg"),
    background: str = Form("white"),
    _: dict = Depends(current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一件服装图")
    jobs = []
    for upload in files:
        raw = await upload.read()
        if not raw:
            continue
        name = upload.filename or f"garment_{uid('g')}.jpg"
        dest = UPLOADS / f"{uid('src')}_{Path(name).name}"
        dest.write_bytes(raw)
        job = create_job(
            {
                "base_id": base_id,
                "audience": audience,
                "garment_type": garment_type,
                "output_format": output_format,
                "background": background,
                "source_name": name,
                "source_url": public_url(dest),
                "title": Path(name).stem,
            }
        )
        jobs.append(job)
    return {"count": len(jobs), "jobs": jobs}


@app.post("/api/jobs/{job_id}/push")
def api_push(job_id: str, body: PushBody, _: dict = Depends(current_user)):
    try:
        job = get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="仅已完成的成品可推送到商品页")
    stores = list_stores()
    store_ids = body.store_ids or []
    if not store_ids:
        store_ids = [s["id"] for s in stores]
    existing = {s["id"]: s for s in stores}
    store_ids = [sid for sid in store_ids if sid in existing]
    if not store_ids:
        raise HTTPException(status_code=400, detail="未找到可推送的店铺，请先在「店铺嵌入」页添加")

    push_body = {
        "sku": body.sku or job.get("sku", ""),
        "product_url": body.product_url or "",
        "product_id": body.product_id or "",
    }
    results = {}
    pushed = []
    push_notes = []
    for sid in store_ids:
        store = existing[sid]
        res = push_store(job, store, push_body)
        results[sid] = res
        if res["ok"]:
            pushed.append(sid)
            push_notes.append(f"{store['name']}: {res['message']}")
        else:
            push_notes.append(f"{store['name']}: 失败 {res['message']}")

    prev = job.get("pushed_stores", [])
    for sid in pushed:
        if sid not in prev:
            prev.append(sid)
    message = "; ".join(push_notes) if push_notes else "未推送任何店铺"
    return patch_job(
        job_id,
        {
            "sku": body.sku or job.get("sku", ""),
            "message": message,
            "pushed": bool(pushed),
            "pushed_stores": prev,
            "push_results": results,
        },
    )


@app.get("/api/health")
def health():
    return {"ok": True, "engine": load_settings().get("engine"), "engine_detail": engine_status()}


@app.get("/api/license")
def license_status():
    status = engine_status()
    return {
        "license_mode": status["license_mode"],
        "commercial_compliant": status["commercial_compliant"],
        "idm_vton_available": status["idm_vton_available"],
        "idm_vton_license": "CC BY-NC-SA 4.0",
        "idm_vton_commercial_use": "禁止",
        "sdxl_available": status["sdxl_available"],
        "gpu_available": status["gpu_available"],
        "current_engine": status["engine"],
        "compliance_advice": status["compliance_advice"],
    }


@app.get("/api/download/{job_id}")
def download(job_id: str, _: dict = Depends(current_user)):
    job = get_job(job_id)
    if not job.get("result_url"):
        raise HTTPException(status_code=404, detail="成品尚未生成")
    path = RESULTS / Path(job["result_url"]).name
    return FileResponse(path, filename=f"{job.get('title') or job_id}{path.suffix}")
