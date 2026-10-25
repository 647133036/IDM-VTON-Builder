"""实物成品图推送适配器。

按店铺 platform 分发到 Shopify Admin API / WordPress REST API，
把任务成品图上传到平台并关联到指定商品。凭证字段：
  - api_key   Shopify: Admin Access Token；WordPress: "用户名:应用密码"
  - shop_url  Shopify: {shop}.myshopify.com（可带协议）；WordPress: 站点根地址
  - product_id 可选；未提供时按任务 SKU 在平台回查商品
"""

import base64
import requests
from pathlib import Path
from typing import Any

from store import RESULTS

SHOPIFY_API_VERSION = "2024-01"


class PushError(RuntimeError):
    pass


def _result_path(job: dict) -> Path:
    if not job.get("result_url"):
        raise PushError("成品尚未生成")
    fname = Path(job["result_url"]).name
    path = RESULTS / fname
    if not path.exists():
        raise PushError(f"成品文件缺失: {fname}")
    return path


def _shopify_shop(store: dict) -> str:
    shop = (store.get("shop_url") or "").strip()
    if not shop:
        raise PushError("店铺缺少 Shopify 域名（shop_url）")
    if not shop.startswith("http"):
        shop = f"https://{shop}"
    return shop.rstrip("/")


def _shopify_token(store: dict) -> str:
    token = (store.get("api_key") or "").strip()
    if not token:
        raise PushError("店铺缺少 Shopify Admin Access Token 或 API Key（api_key）")
    return token


def _wp_site(store: dict) -> str:
    site = (store.get("shop_url") or "").strip()
    if not site:
        raise PushError("店铺缺少 WordPress 站点地址（shop_url）")
    if not site.startswith("http"):
        site = f"https://{site}"
    return site.rstrip("/")


def _wp_auth(store: dict) -> tuple[str, str]:
    cred = (store.get("api_key") or "").strip()
    if ":" not in cred:
        raise PushError("WordPress 店铺的 api_key 需为「用户名:应用密码」格式")
    user, _, password = cred.partition(":")
    if not user or not password:
        raise PushError("WordPress 店铺的 api_key 需为「用户名:应用密码」格式")
    return user, password


def _resolve_shopify_product_id(job: dict, store: dict, body: dict, fix: str, token: str) -> str:
    pid = str((body.get("product_id") or store.get("product_id") or "")).strip()
    if pid:
        if not pid.isdigit():
            raise PushError(f"product_id 应为数字（Shopify 商品 ID），收到: {pid}")
        return pid
    sku = (job.get("sku") or body.get("sku") or "").strip()
    if not sku:
        raise PushError("未提供 product_id，且任务无 SKU 可用于回查商品")
    r = requests.get(
        f"{fix}/admin/api/{SHOPIFY_API_VERSION}/products.json",
        params={"title": sku},
        headers={"X-Shopify-Access-Token": token},
        timeout=30,
    )
    if r.status_code >= 400:
        raise PushError(f"Shopify 查询商品失败 ({r.status_code}): {r.text[:200]}")
    products = r.json().get("products", [])
    if not products:
        raise PushError(f"Shopify 按 SKU/标题未找到商品: {sku}")
    return str(products[0]["id"])


def _resolve_wp_product_id(job: dict, store: dict, body: dict) -> str:
    pid = str((body.get("product_id") or store.get("product_id") or "")).strip()
    if pid:
        if not pid.isdigit():
            raise PushError(f"product_id 应为数字（WooCommerce 商品 ID），收到: {pid}")
        return pid
    sku = (job.get("sku") or body.get("sku") or "").strip()
    if not sku:
        raise PushError("未提供 product_id，且任务无 SKU 可用于回查商品")
    return f"sku:{sku}"


def push_shopify(job: dict, store: dict, body: dict) -> dict[str, Any]:
    fix = _shopify_shop(store)
    token = _shopify_token(store)
    product_id = _resolve_shopify_product_id(job, store, body, fix, token)
    path = _result_path(job)
    url = f"{fix}/admin/api/{SHOPIFY_API_VERSION}/products/{product_id}/images.json"
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = {
        "image": {
            "filename": path.name,
            "attachment": base64.b64encode(path.read_bytes()).decode(),
            "content_type": mime,
        }
    }
    r = requests.post(
        url,
        json=payload,
        headers={"X-Shopify-Access-Token": token},
        timeout=120,
    )
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code >= 400:
        msg = data.get("errors") or data or r.text
        raise PushError(f"Shopify 上传图片失败 ({r.status_code}): {msg}")
    src = data.get("image", {}).get("src", "")
    if not src:
        raise PushError(f"Shopify 未返回图片地址: {data}")
    return {"ok": True, "url": src, "message": f"已上传并关联 Shopify 商品 #{product_id}"}


def push_wordpress(job: dict, store: dict, body: dict) -> dict[str, Any]:
    site = _wp_site(store)
    user, password = _wp_auth(store)
    product_ref = _resolve_wp_product_id(job, store, body)
    path = _result_path(job)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with path.open("rb") as fh:
        r = requests.post(
            f"{site}/wp-json/wp/v2/media",
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
            files={"file": (path.name, fh, mime)},
            auth=(user, password),
            timeout=180,
        )
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code >= 400:
        msg = data.get("message") if isinstance(data, dict) else data
        raise PushError(f"WordPress 上传媒体失败 ({r.status_code}): {msg or r.text[:200]}")
    media_id = data.get("id")
    url = data.get("source_url") or (data.get("guid") or {}).get("rendered", "")
    if not media_id:
        raise PushError(f"WordPress 未返回媒体 ID: {data}")
    if product_ref.isdigit():
        up = requests.post(
            f"{site}/wp-json/wc/v3/products/{product_ref}",
            json={"images": [{"id": media_id}]},
            auth=(user, password),
            timeout=120,
        )
        try:
            up_data = up.json()
        except Exception:
            up_data = {}
        if up.status_code >= 400:
            msg = up_data.get("message") if isinstance(up_data, dict) else up_data
            raise PushError(f"WooCommerce 关联商品失败 ({up.status_code}): {msg or up.text[:200]}")
        return {"ok": True, "url": url, "message": f"已上传媒体并关联 WooCommerce 商品 #{product_ref}"}
    return {"ok": True, "url": url, "message": f"已上传媒体（SKU {product_ref.split(':')[1] or ''}，未关联商品）"}


PUSHERS = {
    "shopify": push_shopify,
    "wordpress": push_wordpress,
}


def push_store(job: dict, store: dict, body: dict) -> dict[str, Any]:
    platform = (store.get("platform") or "other").lower()
    fn = PUSHERS.get(platform)
    if fn is None:
        return {"ok": True, "url": "", "message": "平台未配置适配器，仅打标推送"}
    try:
        return fn(job, store, body)
    except PushError as exc:
        return {"ok": False, "url": "", "message": str(exc)}
    except requests.RequestException as exc:
        return {"ok": False, "url": "", "message": f"网络请求失败: {exc}"}
    except Exception as exc:
        return {"ok": False, "url": "", "message": f"推送失败: {exc}"}