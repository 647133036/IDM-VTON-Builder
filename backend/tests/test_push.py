import pytest
from push import (
    PushError,
    _result_path,
    _shopify_shop,
    _shopify_token,
    _wp_auth,
    _wp_site,
    push_store,
)


@pytest.fixture(autouse=True)
def _mock_dns(monkeypatch):
    import push

    def _safe_getaddrinfo(host, *args, **kwargs):
        return [(0, 0, 0, 0, ("8.8.8.8", 0))]

    monkeypatch.setattr(push.socket, "getaddrinfo", _safe_getaddrinfo)


def _job_file(tmp_path):
    f = tmp_path / "job_x.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (16, 16), (10, 120, 200)).save(f)
    return f


@pytest.fixture()
def done_job(tmp_path, monkeypatch):

    png = tmp_path / "job_done.png"
    from PIL import Image

    Image.new("RGB", (32, 32), (1, 2, 3)).save(png)
    monkeypatch.setattr("push.RESULTS", tmp_path)
    return {
        "id": "job_done",
        "status": "done",
        "result_url": "/files/results/job_done.png",
        "sku": "SKU-001",
    }


def test_push_store_unknown_platform_only_labels():
    job = {"id": "j", "status": "done"}
    store = {"platform": "taobao", "name": "淘宝店"}
    res = push_store(job, store, {})
    assert res["ok"] is True
    assert "tagged only" in res["message"]


def test_push_store_missing_result_returns_failure():
    job = {"id": "j", "status": "done", "result_url": ""}
    store = {"platform": "shopify", "name": "Shop"}
    res = push_store(job, store, {})
    assert res["ok"] is False


def test_result_path_missing_file_raises(done_job, tmp_path):
    with pytest.raises(PushError):
        _result_path({"result_url": "/files/results/nope.png"})


def test_result_path_existing_ok(done_job):
    p = _result_path(done_job)
    assert p.name == "job_done.png"


def test_shopify_shop_normalization():
    assert _shopify_shop({"shop_url": "abc.myshopify.com"}) == "https://abc.myshopify.com"
    assert _shopify_shop({"shop_url": "https://abc.myshopify.com/"}) == "https://abc.myshopify.com"
    with pytest.raises(PushError):
        _shopify_shop({"shop_url": ""})


def test_shopify_token_required():
    assert _shopify_token({"api_key": "tok"}) == "tok"
    with pytest.raises(PushError):
        _shopify_token({"api_key": "  "})


def test_wp_site_normalization():
    assert _wp_site({"shop_url": "example.com"}) == "https://example.com"
    with pytest.raises(PushError):
        _wp_site({"shop_url": ""})


def test_wp_auth_requires_colon():
    assert _wp_auth({"api_key": "user:pass"}) == ("user", "pass")
    with pytest.raises(PushError):
        _wp_auth({"api_key": "missingcolon"})
    with pytest.raises(PushError):
        _wp_auth({"api_key": ":onlypass"})


def test_push_shopify_success(done_job, monkeypatch):
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None, allow_redirects=None):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers

        class R:
            status_code = 200

            def json(self):
                return {"image": {"src": "https://cdn.shop/cloth_a.png"}}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    store = {
        "platform": "shopify",
        "name": "Shop",
        "shop_url": "shop123.myshopify.com",
        "api_key": "shpat_token",
        "product_id": "789",
    }
    res = push_store(done_job, store, {})
    assert res["ok"] is True
    assert "/products/789/images.json" in calls["url"]
    assert calls["headers"]["X-Shopify-Access-Token"] == "shpat_token"


def test_push_shopify_lookup_by_sku_when_no_product_id(done_job, monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None, allow_redirects=None):
        class R:
            status_code = 200

            def json(self):
                return {"products": [{"id": 111}]}

        return R()

    def fake_post(url, json=None, headers=None, timeout=None, allow_redirects=None):
        class R:
            status_code = 200

            def json(self):
                return {"image": {"src": "https://cdn.shop/cloth_b.png"}}

        return R()

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)
    store = {
        "platform": "shopify",
        "name": "Shop",
        "shop_url": "shop123.myshopify.com",
        "api_key": "tok",
        "product_id": "",
    }
    res = push_store(done_job, store, {})
    assert res["ok"] is True
    assert "product #111" in res["message"]


def test_push_shopify_not_found_by_sku_returns_failure(done_job, monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None, allow_redirects=None):
        class R:
            status_code = 200

            def json(self):
                return {"products": []}

        return R()

    monkeypatch.setattr("requests.get", fake_get)
    store = {
        "platform": "shopify",
        "name": "Shop",
        "shop_url": "shop123.myshopify.com",
        "api_key": "tok",
        "product_id": "",
    }
    res = push_store(done_job, store, {})
    assert res["ok"] is False
    assert "no product found" in res["message"]


def test_push_wordpress_success(done_job, monkeypatch):
    calls = {}

    def fake_post(url, headers=None, files=None, json=None, auth=None, timeout=None, allow_redirects=None):
        calls.setdefault("urls", []).append(url)
        calls["auth"] = auth

        class R:
            status_code = 200

            def json(self):
                return {"id": 42, "source_url": "https://site/img.png"}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    store = {
        "platform": "wordpress",
        "name": "WP",
        "shop_url": "https://shop.example",
        "api_key": "user:app",
        "product_id": "77",
    }
    res = push_store(done_job, store, {})
    assert res["ok"] is True
    assert "linked WooCommerce product #77" in res["message"]
    assert calls["auth"] == ("user", "app")


def test_push_wordpress_media_failure(done_job, monkeypatch):
    def fake_post(url, headers=None, files=None, auth=None, timeout=None, json=None, allow_redirects=None):
        class R:
            status_code = 401
            text = "unauthorized"

            def json(self):
                return {"message": "wrong password"}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    store = {
        "platform": "wordpress",
        "name": "WP",
        "shop_url": "https://shop.example",
        "api_key": "user:bad",
        "product_id": "1",
    }
    res = push_store(done_job, store, {})
    assert res["ok"] is False
    assert "wrong password" in res["message"]


def test_push_wordpress_without_product_id_uploads_only(done_job, monkeypatch):
    def fake_post(url, headers=None, files=None, auth=None, timeout=None, json=None, allow_redirects=None):
        class R:
            status_code = 200

            def json(self):
                return {"id": 99, "source_url": "https://site/img.png"}

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    store = {
        "platform": "wordpress",
        "name": "WP",
        "shop_url": "https://shop.example",
        "api_key": "user:app",
        "product_id": "",
    }
    res = push_store(done_job, store, {"sku": "SKU-001"})
    assert res["ok"] is True
    assert "no product linked" in res["message"]


def test_push_network_error_caught(done_job, monkeypatch):
    def fake_post(url, **kwargs):
        import requests

        raise requests.ConnectionError("boom")

    monkeypatch.setattr("requests.post", fake_post)
    store = {
        "platform": "shopify",
        "name": "Shop",
        "shop_url": "shop.myshopify.com",
        "api_key": "tok",
        "product_id": "1",
    }
    res = push_store(done_job, store, {})
    assert res["ok"] is False
    assert "Network request failed" in res["message"]


def test_ssrf_http_scheme_rejected():
    with pytest.raises(PushError, match="HTTPS"):
        _wp_site({"shop_url": "http://example.com"})


def test_ssrf_shopify_suffix_enforced():
    with pytest.raises(PushError, match=r"myshopify\.com"):
        _shopify_shop({"shop_url": "evil.com"})


def test_ssrf_internal_ip_rejected(monkeypatch):
    import push

    def _fake_getaddrinfo(host, *args, **kwargs):
        return [(0, 0, 0, 0, ("127.0.0.1", 0))]

    monkeypatch.setattr(push.socket, "getaddrinfo", _fake_getaddrinfo)
    with pytest.raises(PushError, match="private/reserved"):
        push._check_host_safe("evil.com")
