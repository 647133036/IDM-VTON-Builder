import io
import re
import uuid
from pathlib import Path
from typing import Literal

from auth import USERS, current_user, issue_token, require_admin
from engine import engine_status
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from push import push_store
from pydantic import BaseModel, Field
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
    list_bases,
    list_ethnicities,
    list_jobs,
    list_stores,
    patch_job,
    public_url,
    seed_bases_if_empty,
    stats,
    uid,
    update_base,
    update_ethnicity,
    update_settings,
    update_store,
)
from store import (
    settings as load_settings,
)
from worker import start_worker

app = FastAPI(title="Atelier Try-On", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

seed_bases_if_empty()
start_worker()

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# 像素炸弹上限:解压后像素数不得超过此值。
_MAX_IMAGE_PIXELS = 64_000_000
# 单次批量任务上限:张数 + 累计体积。
_MAX_BATCH_FILES = 50
_MAX_BATCH_BYTES = 200 * 1024 * 1024
_PUBLIC_DIRS = {"uploads", "results", "bases"}


def _resolve_public_file(path: str) -> Path | None:
    rel = Path(path)
    parts = rel.parts
    if not parts:
        return None
    top = parts[0]
    if top not in _PUBLIC_DIRS:
        return None
    target = (DATA / rel).resolve()
    data_root = DATA.resolve()
    if target != data_root / top and data_root / top not in target.parents:
        return None
    if not target.is_file():
        if top == "uploads":
            found = find_in_dir(UPLOADS, rel.name)
            if found:
                return found
        return None
    return target


def _safe_name(value: str) -> str:
    name = value.strip()
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid name")
    if not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid name: use letters, digits, dot, dash, underscore only")
    return name


def _safe_category(value: str) -> str:
    # 分类名禁止使用保留字 default,避免与"未分类"根目录冲突(P2-5)。
    name = _safe_name(value)
    if name == "default":
        raise HTTPException(status_code=400, detail="Category 'default' is reserved")
    return name


def _category_dir(category: str) -> Path:
    # 空/缺省/保留字 default 一律落在"未分类"根目录,不再创建 default 子目录。
    if not category or category == "default":
        return UPLOADS
    return UPLOADS / _safe_category(category)


def _validate_image_bytes(raw: bytes) -> tuple[bytes, str]:
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            # 按像素数上限显式拒绝,防止解压炸弹(H-10)。
            if width * height > _MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail=f"Too many pixels (max {_MAX_IMAGE_PIXELS})")
            fmt = (img.format or "").upper()
            img.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
    ext = ext_map.get(fmt)
    if ext is None:
        raise HTTPException(status_code=400, detail=f"Unsupported image format {fmt}; use PNG, JPEG, or WEBP")
    if ext not in _IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image format; use PNG, JPEG, or WEBP")
    return raw, ext


@app.get("/files/{path:path}")
def serve_file(path: str):
    target = _resolve_public_file(path)
    if target is None:
        raise HTTPException(status_code=404, detail="File not found")
    suffix = target.suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="File not found")
    headers = {"X-Content-Type-Options": "nosniff"}
    return FileResponse(target, headers=headers)


class LoginBody(BaseModel):
    username: str
    password: str


class SettingsBody(BaseModel):
    output_long_side: int | None = Field(default=None, ge=256, le=4096)
    background: Literal["white", "transparent"] | None = None
    engine: str | None = Field(default=None, max_length=64)
    store_name: str | None = Field(default=None, max_length=120)
    webhook_url: str | None = Field(default=None, max_length=512)


class BasePatch(BaseModel):
    name: str | None = None
    audience: str | None = None
    age_band: str | None = None
    ethnicity: str | None = None
    pose: str | None = None
    notes: str | None = None


class BaseBatchPatch(BaseModel):
    ids: list[str]
    patch: dict[str, str | None] = {}


class EthnicityBody(BaseModel):
    value: str
    label: str


class EthnicityPatch(BaseModel):
    label: str | None = None


class CategoryBody(BaseModel):
    name: str


class StoreBody(BaseModel):
    name: str
    platform: str | None = "other"
    shop_url: str | None = ""
    webhook_url: str | None = ""
    api_key: str | None = ""
    product_id: str | None = ""
    default_sku: str | None = ""


class StorePatch(BaseModel):
    name: str | None = None
    platform: str | None = None
    shop_url: str | None = None
    webhook_url: str | None = None
    api_key: str | None = None
    product_id: str | None = None
    default_sku: str | None = None


class PushBody(BaseModel):
    sku: str | None = ""
    product_url: str | None = ""
    product_id: str | None = ""
    store_ids: list[str] | None = []


@app.post("/api/auth/login")
def login(body: LoginBody):
    user = USERS.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
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
def api_update_settings(body: SettingsBody, _: dict = Depends(require_admin)):
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
    _: dict = Depends(require_admin),
):
    if ethnicity not in {e["value"] for e in list_ethnicities()}:
        raise HTTPException(status_code=400, detail=f"Unknown ethnicity tag: {ethnicity}")
    if audience not in _AUDIENCE_SET - {"auto"}:
        raise HTTPException(status_code=400, detail=f"Invalid audience: {audience}")
    if age_band not in {"adult", "kids", "baby"}:
        raise HTTPException(status_code=400, detail=f"Invalid age_band: {age_band}")
    content = await file.read()
    _, ext = _validate_image_bytes(content)
    base_id = uid("base")
    filename = f"{base_id}{ext}"
    dest = BASES / filename
    dest.write_bytes(content)
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
def api_patch_base(base_id: str, body: BasePatch, _: dict = Depends(require_admin)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    if "ethnicity" in patch and patch["ethnicity"] not in {e["value"] for e in list_ethnicities()}:
        raise HTTPException(status_code=400, detail=f"Unknown ethnicity tag: {patch['ethnicity']}")
    try:
        return update_base(base_id, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Base model not found") from exc


@app.post("/api/bases/batch/patch")
def api_patch_bases_batch(body: BaseBatchPatch, _: dict = Depends(require_admin)):
    patch = {k: v for k, v in body.patch.items() if v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="No updatable fields provided")
    allowed = {"name", "audience", "age_band", "ethnicity", "pose", "notes"}
    patch = {k: v for k, v in patch.items() if k in allowed}
    if "ethnicity" in patch and patch["ethnicity"] not in {e["value"] for e in list_ethnicities()}:
        raise HTTPException(status_code=400, detail=f"Unknown ethnicity tag: {patch['ethnicity']}")
    updated = []
    for base_id in body.ids:
        try:
            updated.append(update_base(base_id, dict(patch)))
        except KeyError:
            continue
    return {"ok": True, "count": len(updated), "bases": updated}


@app.delete("/api/bases/{base_id}")
def api_delete_base(base_id: str, _: dict = Depends(require_admin)):
    try:
        delete_base(base_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Base model not found") from exc
    return {"ok": True, "id": base_id}


@app.get("/api/ethnicities")
def api_ethnicities(_: dict = Depends(current_user)):
    return list_ethnicities()


@app.post("/api/ethnicities")
def api_add_ethnicity(body: EthnicityBody, _: dict = Depends(require_admin)):
    value = body.value.strip()
    label = body.label.strip()
    if not value or not label:
        raise HTTPException(status_code=400, detail="Ethnicity value and label must not be empty")
    # value 会渲染进 HTML 属性(NEW-M2),必须限定为安全的小写 slug,禁止引号/尖括号/空格。
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value):
        raise HTTPException(
            status_code=400,
            detail="Ethnicity value must be a lowercase slug: letters, digits, underscore, max 32 chars",
        )
    try:
        return add_ethnicity({"value": value, "label": label})
    except KeyError as exc:
        raise HTTPException(status_code=400, detail="Ethnicity tag already exists") from exc


@app.patch("/api/ethnicities/{value}")
def api_patch_ethnicity(value: str, body: EthnicityPatch, _: dict = Depends(require_admin)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return update_ethnicity(value, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ethnicity tag not found") from exc


@app.delete("/api/ethnicities/{value}")
def api_delete_ethnicity(value: str, _: dict = Depends(require_admin)):
    try:
        delete_ethnicity(value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ethnicity tag not found") from exc
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
    base = base.removesuffix("/api")
    return base


def _mask_store(item: dict, origin: str) -> dict:
    key = item.get("api_key", "")
    item["api_key"] = ""
    item["has_api_key"] = bool(key)
    if key:
        # 短 key 整体隐藏;≥12 位才暴露首尾各 3 位(NEW-L4,避免 9 位 key 泄露 2/3)。
        item["api_key_masked"] = key[:3] + "***" + key[-3:] if len(key) >= 12 else "***"
    item["embed_url"] = embed_url(origin, item["id"])
    return item


@app.get("/api/stores")
def api_stores(request: Request, _: dict = Depends(current_user)):
    origin = _origin(request)
    return [_mask_store(dict(s), origin) for s in list_stores()]


@app.post("/api/stores")
def api_add_store(body: StoreBody, request: Request, _: dict = Depends(require_admin)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Store name must not be empty")
    meta = {
        "id": uid("store"),
        "name": name,
        "platform": body.platform or "other",
        "shop_url": body.shop_url or "",
        "webhook_url": body.webhook_url or "",
        "api_key": body.api_key or "",
        "product_id": body.product_id or "",
        "default_sku": body.default_sku or "",
        "created_at": int(__import__("time").time()),
        "active": True,
    }
    created = add_store(meta)
    return _mask_store(created, _origin(request))


@app.patch("/api/stores/{store_id}")
def api_patch_store(store_id: str, body: StorePatch, request: Request, _: dict = Depends(require_admin)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return _mask_store(update_store(store_id, patch), _origin(request))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Store not found") from exc


@app.delete("/api/stores/{store_id}")
def api_delete_store(store_id: str, _: dict = Depends(require_admin)):
    try:
        delete_store(store_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Store not found") from exc
    return {"ok": True, "id": store_id}


@app.get("/api/jobs")
def api_jobs(_: dict = Depends(current_user)):
    return list_jobs()


@app.get("/api/uploads")
def api_list_uploads(category: str | None = None, _: dict = Depends(current_user)):
    items = []
    if category:
        # 统一用 _category_dir:default 映射到 UPLOADS 根目录(P2-5),
        # 否则 GET /api/uploads?category=default 永远返回 []。
        cat_dir = _category_dir(category)
        cat = "default" if category == "default" else _safe_category(category)
        if cat_dir.exists():
            for p in sorted(cat_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                    items.append(
                        {
                            "name": p.name,
                            "url": public_url(p),
                            "size": p.stat().st_size,
                            "mtime": int(p.stat().st_mtime),
                            "category": cat,
                        }
                    )
    else:
        for p in sorted(UPLOADS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                items.append(
                    {
                        "name": p.name,
                        "url": public_url(p),
                        "size": p.stat().st_size,
                        "mtime": int(p.stat().st_mtime),
                        "category": "default",
                    }
                )
            elif p.is_dir():
                for f in sorted(p.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES:
                        items.append(
                            {
                                "name": f.name,
                                "url": public_url(f),
                                "size": f.stat().st_size,
                                "mtime": int(f.stat().st_mtime),
                                "category": p.name,
                            }
                        )
    return items


@app.delete("/api/uploads/{filename}")
def api_delete_upload(filename: str, _: dict = Depends(require_admin)):
    safe_name = _safe_name(filename)
    p = UPLOADS / safe_name
    if not p.is_file():
        for d in UPLOADS.iterdir():
            if d.is_dir():
                candidate = d / safe_name
                if candidate.is_file():
                    p = candidate
                    break
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if p.suffix.lower() not in _IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only image files can be deleted here")
    trash_dest = TRASH / f"upload_{p.parent.name}_{safe_name}"
    if trash_dest.exists():
        trash_dest.unlink()
    p.replace(trash_dest)
    return {"ok": True, "filename": safe_name}


class MoveBody(BaseModel):
    category: str = "default"


@app.post("/api/uploads/{filename}/move")
def api_move_upload(filename: str, body: MoveBody, _: dict = Depends(require_admin)):
    safe_name = _safe_name(filename)
    p = UPLOADS / safe_name
    if not p.is_file():
        for d in UPLOADS.iterdir():
            if d.is_dir():
                candidate = d / safe_name
                if candidate.is_file():
                    p = candidate
                    break
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if p.suffix.lower() not in _IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only image files can be moved here")
    category = body.category or "default"
    cat_dir = _category_dir(category)
    # 目标目录就是源目录时为无操作;否则会把根目录文件改名成 upload_xxx_1(P2-5)。
    if cat_dir.resolve() == p.parent.resolve():
        return {"ok": True, "filename": p.name, "category": "default" if category == "default" else category}
    cat_dir.mkdir(parents=True, exist_ok=True)
    dest = cat_dir / safe_name
    i = 1
    while dest.exists():
        dest = cat_dir / f"{Path(safe_name).stem}_{i}{Path(safe_name).suffix}"
        i += 1
    p.replace(dest)
    return {"ok": True, "filename": dest.name, "category": category}


@app.post("/api/uploads/categories")
def api_add_upload_category(body: CategoryBody, _: dict = Depends(require_admin)):
    name = _safe_name(body.name)
    if name == "default":
        raise HTTPException(status_code=400, detail="Category 'default' is reserved")
    cat_dir = UPLOADS / name
    cat_dir.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "category": name}


@app.patch("/api/uploads/categories/{category}")
def api_rename_upload_category(category: str, body: CategoryBody, _: dict = Depends(require_admin)):
    src_dir = UPLOADS / _safe_name(category)
    new_name = _safe_name(body.name)
    if new_name == "default":
        raise HTTPException(status_code=400, detail="Category 'default' is reserved")
    if not src_dir.is_dir():
        raise HTTPException(status_code=404, detail="Category not found")
    if new_name == src_dir.name:
        return {"ok": True, "category": new_name}
    if (UPLOADS / new_name).exists():
        raise HTTPException(status_code=400, detail=f"Category already exists: {new_name}")
    src_dir.rename(UPLOADS / new_name)
    return {"ok": True, "category": new_name}


@app.delete("/api/uploads/categories/{category}")
def api_delete_upload_category(category: str, _: dict = Depends(require_admin)):
    cat_dir = UPLOADS / _safe_name(category)
    if not cat_dir.is_dir():
        raise HTTPException(status_code=404, detail="Category not found")
    moved = 0
    for p in cat_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
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
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            categories["default"] += 1
        elif p.is_dir():
            count = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES)
            categories[p.name] = count
    return categories


@app.get("/api/uploads/{category}/{filename}")
def api_get_upload(category: str, filename: str, _: dict = Depends(current_user)):
    safe_category = _safe_name(category)
    safe_name = _safe_name(filename)
    # 与 list/move 统一:default 映射到 UPLOADS 根目录(P2-5)。
    p = _category_dir(safe_category) / safe_name
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if p.suffix.lower() not in _IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Image not found")
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
    content = await file.read()
    _, ext = _validate_image_bytes(content)
    safe_name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    cat = category or "default"
    cat_dir = _category_dir(cat)
    cat_dir.mkdir(parents=True, exist_ok=True)
    dest = cat_dir / safe_name
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
        raise HTTPException(status_code=404, detail="Job not found") from exc


_AUDIENCE_SET = {"women", "men", "girls", "boys", "baby", "auto"}
_GARMENT_SET = {"upper", "lower", "dress", "onesie", "auto"}
_BG_SET = {"white", "transparent"}
_FMT_SET = {"jpg", "jpeg", "png"}


def _validate_batch_params(audience: str, garment_type: str, background: str, output_format: str) -> None:
    if audience not in _AUDIENCE_SET:
        raise HTTPException(status_code=400, detail=f"Invalid audience: {audience}")
    if garment_type not in _GARMENT_SET:
        raise HTTPException(status_code=400, detail=f"Invalid garment_type: {garment_type}")
    if background not in _BG_SET:
        raise HTTPException(status_code=400, detail=f"Invalid background: {background}")
    if output_format not in _FMT_SET:
        raise HTTPException(status_code=400, detail=f"Invalid output_format: {output_format}")


@app.post("/api/jobs/batch")
async def api_batch(
    files: list[UploadFile] = File(...),
    base_id: str = Form(...),
    audience: str = Form("auto"),
    garment_type: str = Form("auto"),
    output_format: str = Form("jpg"),
    background: str = Form("white"),
    bound_store_id: str = Form(""),
    _: dict = Depends(current_user),
):
    _validate_batch_params(audience, garment_type, background, output_format)
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one garment image")
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files (max {_MAX_BATCH_FILES})")
    bases = list_bases()
    if not any(b["id"] == base_id for b in bases):
        raise HTTPException(status_code=400, detail="Base model not found")
    jobs = []
    prepared = []
    total_bytes = 0
    for upload in files:
        raw = await upload.read()
        if not raw:
            continue
        # 累计体积校验:防止超大批量拖垮磁盘/CPU。
        total_bytes += len(raw)
        if total_bytes > _MAX_BATCH_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Batch too large (max {_MAX_BATCH_BYTES // (1024 * 1024)}MB)",
            )
        _, ext = _validate_image_bytes(raw)
        display_name = upload.filename or f"garment{ext}"
        prepared.append((raw, ext, display_name))
    if not prepared:
        raise HTTPException(status_code=400, detail="Please upload at least one valid garment image")
    for raw, ext, display_name in prepared:
        src_name = f"{uid('src')}{ext}"
        dest = UPLOADS / src_name
        dest.write_bytes(raw)
        job = create_job(
            {
                "base_id": base_id,
                "audience": audience,
                "garment_type": garment_type,
                "output_format": output_format,
                "background": background,
                "source_name": display_name,
                "source_url": public_url(dest),
                "title": Path(display_name).stem,
                "bound_store_id": bound_store_id or "",
            }
        )
        jobs.append(job)
    return {"count": len(jobs), "jobs": jobs}


class ExportCsvBody(BaseModel):
    job_ids: list[str]


@app.post("/api/jobs/export-csv")
def api_export_csv(body: ExportCsvBody, _: dict = Depends(current_user)):
    """导出 ZIP:CSV 清单 + 已完成任务的成品图,供离线修图后批量回传推送。"""
    from export_csv import export_jobs_csv

    if not body.job_ids:
        raise HTTPException(status_code=400, detail="Please select at least one job")
    data = export_jobs_csv(body.job_ids)
    if not data:
        raise HTTPException(status_code=400, detail="No completed jobs found to export")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=tryon_export.zip"},
    )


@app.post("/api/stores/{store_id}/batch-push")
async def api_batch_push(
    store_id: str,
    csv_file: UploadFile = File(...),
    files: list[UploadFile] = File(default=[]),
    _: dict = Depends(require_admin),
):
    """CSV + 图片包批量推送:解析 CSV,逐行匹配替换图,推送到当前店铺。

    流程:
    1. 解析 CSV(manifest.csv 格式),校验列名
    2. 逐行:store_id 缺失填当前店铺;image_filename 在上传包中存在用替换图,
       否则用原成品图;都不存在跳过标注「图片缺失」
    3. product_id/sku 缺失回退店铺 default_product_id / default_sku
    4. 逐行调 push_store,返回 [{job_id, ok, message, url}]
    """
    import csv as csv_mod

    from push import push_store

    stores = {s["id"]: s for s in list_stores()}
    if store_id not in stores:
        raise HTTPException(status_code=404, detail="Store not found")
    store = stores[store_id]

    raw_csv = (await csv_file.read()).decode("utf-8-sig")
    reader = csv_mod.DictReader(io.StringIO(raw_csv))
    required = {"job_id", "image_filename"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {','.join(sorted(missing))}. Expected: {','.join(sorted(required))}",
        )

    # 上传图片包:filename -> bytes
    uploaded: dict[str, bytes] = {}
    for f in files:
        data = await f.read()
        if data:
            uploaded[f.filename or ""] = data

    results = []
    for row in reader:
        jid = (row.get("job_id") or "").strip()
        img_name = (row.get("image_filename") or "").strip()
        if not jid:
            results.append({"job_id": "", "ok": False, "message": "Empty job_id", "url": ""})
            continue

        try:
            job = get_job(jid)
        except KeyError:
            results.append({"job_id": jid, "ok": False, "message": "Job not found", "url": ""})
            continue

        # 替换图优先:上传包有同名文件 → 写到 RESULTS 临时位置,push_store 用它
        result_url = job.get("result_url", "")
        if img_name and img_name in uploaded:
            replaced_name = f"{jid}_replaced.jpg"
            replaced_path = RESULTS / replaced_name
            replaced_path.write_bytes(uploaded[img_name])
            result_url = f"/files/results/{replaced_name}"
        elif not result_url or not (RESULTS / result_url.split("/")[-1]).exists():
            results.append({"job_id": jid, "ok": False, "message": "图片缺失", "url": ""})
            continue

        # SKU/product_id 回退链:CSV 行 > 店铺默认
        sku = (row.get("sku") or "").strip() or store.get("default_sku", "")
        product_id = (row.get("product_id") or "").strip() or store.get("product_id", "")

        # 构造临时 job dict(不污染原 job),用替换图路径
        push_job = dict(job)
        push_job["result_url"] = result_url
        push_body = {"sku": sku, "product_id": product_id, "product_url": ""}
        res = push_store(push_job, store, push_body)
        results.append({"job_id": jid, "ok": res["ok"], "message": res["message"], "url": res.get("url", "")})

    return {"results": results, "total": len(results), "ok": sum(1 for r in results if r["ok"])}


@app.post("/api/jobs/{job_id}/push")
def api_push(job_id: str, body: PushBody, _: dict = Depends(require_admin)):
    try:
        job = get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Only completed results can be pushed to a product page")
    stores = list_stores()
    store_ids = body.store_ids or []
    if not store_ids:
        # 回退链:显式传入 > job 绑定店铺 > 全部店铺
        bound = job.get("bound_store_id", "")
        store_ids = [bound] if bound else [s["id"] for s in stores]
    existing = {s["id"]: s for s in stores}
    store_ids = [sid for sid in store_ids if sid in existing]
    if not store_ids:
        raise HTTPException(status_code=400, detail="No pushable store found; add one on the Store Embed page first")

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
        # 每店 SKU/product_id 回退:body 显式 > 店铺默认 > 空
        per_sku = push_body["sku"] or store.get("default_sku", "")
        per_pid = push_body["product_id"] or store.get("product_id", "")
        per_body = {**push_body, "sku": per_sku, "product_id": per_pid}
        res = push_store(job, store, per_body)
        results[sid] = {**res, "product_id": per_pid, "sku": per_sku}
        if res["ok"]:
            pushed.append(sid)
            push_notes.append(f"{store['name']}: {res['message']}")
        else:
            push_notes.append(f"{store['name']}: failed {res['message']}")

    prev = job.get("pushed_stores", [])
    for sid in pushed:
        if sid not in prev:
            prev.append(sid)
    message = "; ".join(push_notes) if push_notes else "No store pushed"
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
    detail = engine_status()
    # 不向前端暴露本地绝对路径(weights_dir/idm_vton_dir/sdxl_dir)。
    for _k in ("weights_dir", "idm_vton_dir", "sdxl_dir"):
        detail.pop(_k, None)
    return {"ok": True, "engine": load_settings().get("engine"), "engine_detail": detail}


@app.get("/api/license")
def license_status():
    status = engine_status()
    return {
        "license_mode": status["license_mode"],
        "commercial_compliant": status["commercial_compliant"],
        "idm_vton_available": status["idm_vton_available"],
        "idm_vton_license": "CC BY-NC-SA 4.0",
        "idm_vton_commercial_use": "Forbidden",
        "sdxl_available": status["sdxl_available"],
        "ip_adapter_available": status["ip_adapter_available"],
        "gpu_available": status["gpu_available"],
        "current_engine": status["engine"],
        "compliance_advice": status["compliance_advice"],
    }


@app.get("/api/download/{job_id}")
def download(job_id: str, _: dict = Depends(current_user)):
    try:
        job = get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if not job.get("result_url"):
        raise HTTPException(status_code=404, detail="Result not generated yet")
    path = RESULTS / Path(job["result_url"]).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(path, filename=f"{job.get('title') or job_id}{path.suffix}")
