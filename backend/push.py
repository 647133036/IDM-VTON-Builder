"""Final image push adapters.

Dispatch by store platform to Shopify Admin API / WordPress REST API:
upload the job result image to the platform and link it to the target product.
Credentials:
  - api_key   Shopify: Admin Access Token; WordPress: "username:app-password"
  - shop_url  Shopify: {shop}.myshopify.com (protocol optional); WordPress: site root URL
  - product_id optional; if absent, look up the product by job SKU on the platform
"""

import base64
import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from store import RESULTS

SHOPIFY_API_VERSION = "2024-10"


class PushError(RuntimeError):
    pass


def _check_host_safe(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise PushError(f"Cannot resolve store hostname: {exc}") from exc
    if not infos:
        raise PushError("Store hostname resolved to no addresses")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # 仅允许全局可路由单播:一并拒绝内网/回环/链路本地/保留/CGNAT(100.64/10)/
        # 组播(224/4)/未指定,拦截绝大多数 SSRF 与指向内网的重绑定解析(H-03)。
        # 注意:IPv4 组播 is_global 在 CPython 返回 True,需显式判 is_multicast。
        # 局限:此检查仅校验校验时刻的解析结果;后续 HTTP 请求会再次解析,
        # 若攻击者在此窗口内把主机名改解析到内网地址,本检查无法拦截(TOCTOU)。
        if not ip.is_global or ip.is_multicast:
            raise PushError("Store URL resolves to a non-global IP address")


def _validate_shop_url(raw: str, require_suffix: str | None = None) -> str:
    shop = raw.strip()
    if not shop:
        raise PushError("Store URL is empty")
    if not shop.startswith("http"):
        shop = f"https://{shop}"
    parsed = urlparse(shop)
    if parsed.scheme != "https":
        raise PushError("Store URL must use HTTPS")
    host = parsed.hostname or ""
    if not host:
        raise PushError("Store URL has no hostname")
    if require_suffix and not host.endswith(require_suffix):
        raise PushError(f"Store URL must end with {require_suffix}")
    _check_host_safe(host)
    return f"https://{host}{parsed.path}".rstrip("/")


def _result_path(job: dict) -> Path:
    if not job.get("result_url"):
        raise PushError("Result not generated yet")
    fname = Path(job["result_url"]).name
    path = RESULTS / fname
    if not path.exists():
        raise PushError(f"Result file missing: {fname}")
    return path


def _shopify_shop(store: dict) -> str:
    shop = (store.get("shop_url") or "").strip()
    if not shop:
        raise PushError("Store missing Shopify domain (shop_url)")
    return _validate_shop_url(shop, require_suffix=".myshopify.com")


def _shopify_token(store: dict) -> str:
    token = (store.get("api_key") or "").strip()
    if not token:
        raise PushError("Store missing Shopify Admin Access Token or API key (api_key)")
    return token


def _wp_site(store: dict) -> str:
    site = (store.get("shop_url") or "").strip()
    if not site:
        raise PushError("Store missing WordPress site URL (shop_url)")
    return _validate_shop_url(site)


def _wp_auth(store: dict) -> tuple[str, str]:
    cred = (store.get("api_key") or "").strip()
    if ":" not in cred:
        raise PushError("WordPress store api_key must be in 'username:app-password' format")
    user, _, password = cred.partition(":")
    if not user or not password:
        raise PushError("WordPress store api_key must be in 'username:app-password' format")
    return user, password


def _resolve_shopify_product_id(job: dict, store: dict, body: dict, fix: str, token: str) -> str:
    pid = str(body.get("product_id") or store.get("product_id") or "").strip()
    if pid:
        if not pid.isdigit():
            raise PushError(f"product_id must be numeric (Shopify product ID), got: {pid}")
        return pid
    sku = (job.get("sku") or body.get("sku") or "").strip()
    if not sku:
        raise PushError("No product_id provided and job has no SKU to look up")
    r = requests.get(
        f"{fix}/admin/api/{SHOPIFY_API_VERSION}/products.json",
        params={"title": sku},
        headers={"X-Shopify-Access-Token": token},
        timeout=30,
        allow_redirects=False,
    )
    if r.status_code >= 400:
        raise PushError(f"Shopify product lookup failed ({r.status_code}): {r.text[:200]}")
    products = r.json().get("products", [])
    if not products:
        raise PushError(f"Shopify: no product found by SKU/title: {sku}")
    return str(products[0]["id"])


def _resolve_wp_product_id(job: dict, store: dict, body: dict) -> str:
    pid = str(body.get("product_id") or store.get("product_id") or "").strip()
    if pid:
        if not pid.isdigit():
            raise PushError(f"product_id must be numeric (WooCommerce product ID), got: {pid}")
        return pid
    sku = (job.get("sku") or body.get("sku") or "").strip()
    if not sku:
        raise PushError("No product_id provided and job has no SKU to look up")
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
        allow_redirects=False,
    )
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        data = {}
    if r.status_code >= 400:
        msg = data.get("errors") or data or r.text
        raise PushError(f"Shopify image upload failed ({r.status_code}): {msg}")
    src = data.get("image", {}).get("src", "")
    if not src:
        raise PushError(f"Shopify did not return image URL: {data}")
    return {"ok": True, "url": src, "message": f"Uploaded and linked Shopify product #{product_id}"}


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
            allow_redirects=False,
        )
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        data = {}
    if r.status_code >= 400:
        msg = data.get("message") if isinstance(data, dict) else data
        raise PushError(f"WordPress media upload failed ({r.status_code}): {msg or r.text[:200]}")
    media_id = data.get("id")
    url = data.get("source_url") or (data.get("guid") or {}).get("rendered", "")
    if not media_id:
        raise PushError(f"WordPress did not return media ID: {data}")
    if product_ref.isdigit():
        up = requests.post(
            f"{site}/wp-json/wc/v3/products/{product_ref}",
            json={"images": [{"id": media_id}]},
            auth=(user, password),
            timeout=120,
            allow_redirects=False,
        )
        try:
            up_data = up.json()
        except Exception:  # noqa: BLE001
            up_data = {}
        if up.status_code >= 400:
            msg = up_data.get("message") if isinstance(up_data, dict) else up_data
            raise PushError(f"WooCommerce product link failed ({up.status_code}): {msg or up.text[:200]}")
        return {"ok": True, "url": url, "message": f"Uploaded media and linked WooCommerce product #{product_ref}"}
    return {
        "ok": True,
        "url": url,
        "message": f"Uploaded media (SKU {product_ref.split(':')[1] or ''}, no product linked)",
    }


PUSHERS = {
    "shopify": push_shopify,
    "wordpress": push_wordpress,
}


def push_store(job: dict, store: dict, body: dict) -> dict[str, Any]:
    platform = (store.get("platform") or "other").lower()
    fn = PUSHERS.get(platform)
    if fn is None:
        return {"ok": True, "url": "", "message": "No adapter configured for platform; push tagged only"}
    try:
        return fn(job, store, body)
    except PushError as exc:
        return {"ok": False, "url": "", "message": str(exc)}
    except requests.RequestException as exc:
        return {"ok": False, "url": "", "message": f"Network request failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": "", "message": f"Push failed: {exc}"}
