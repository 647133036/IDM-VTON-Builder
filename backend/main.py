from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import USERS, current_user, issue_token
from engine import guess_audience, guess_garment_type
from store import (
    BASES,
    DATA,
    RESULTS,
    UPLOADS,
    add_base,
    create_job,
    get_job,
    delete_base,
    list_bases,
    list_jobs,
    patch_job,
    public_url,
    seed_bases_if_empty,
    settings as load_settings,
    stats,
    uid,
    update_base,
    update_settings,
)
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
app.mount("/files", StaticFiles(directory=str(DATA)), name="files")


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
    notes: Optional[str] = None
    active: Optional[bool] = None


class PushBody(BaseModel):
    sku: Optional[str] = ""
    product_url: Optional[str] = ""


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
    notes: str = Form(""),
    file: UploadFile = File(...),
    _: dict = Depends(current_user),
):
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
    try:
        return update_base(base_id, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="底座不存在") from exc


@app.delete("/api/bases/{base_id}")
def api_delete_base(base_id: str, _: dict = Depends(current_user)):
    try:
        delete_base(base_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="底座不存在") from exc
    return {"ok": True, "id": base_id}


@app.get("/api/jobs")
def api_jobs(_: dict = Depends(current_user)):
    return list_jobs()


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
        resolved_audience = audience if audience != "auto" else audience
        resolved_type = garment_type if garment_type != "auto" else garment_type
        job = create_job(
            {
                "base_id": base_id,
                "audience": resolved_audience,
                "garment_type": resolved_type,
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
    return patch_job(
        job_id,
        {
            "sku": body.sku or job.get("sku", ""),
            "message": f"已推送到商品页 {body.product_url or job.get('title', '')}",
            "pushed": True,
        },
    )


@app.get("/api/health")
def health():
    return {"ok": True, "engine": load_settings().get("engine")}


@app.get("/api/download/{job_id}")
def download(job_id: str, _: dict = Depends(current_user)):
    job = get_job(job_id)
    if not job.get("result_url"):
        raise HTTPException(status_code=404, detail="成品尚未生成")
    path = RESULTS / Path(job["result_url"]).name
    return FileResponse(path, filename=f"{job.get('title') or job_id}{path.suffix}")
