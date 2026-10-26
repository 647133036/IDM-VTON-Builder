import pytest
from PIL import Image


def _png_bytes(color=(120, 40, 200), size=(200, 300)):
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_login_success(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    body = r.json()
    assert body["token"]
    assert body["role"] == "admin"


def test_login_wrong_password(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/api/me").status_code == 401


def test_me_with_token(client, auth_headers):
    r = client.get("/api/me", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["sub"] == "admin"


def test_health_no_auth(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_license_no_auth(client):
    r = client.get("/api/license")
    assert r.status_code == 200
    assert r.json()["idm_vton_license"] == "CC BY-NC-SA 4.0"


def test_ethnicities_list(client, auth_headers):
    r = client.get("/api/ethnicities", headers=auth_headers)
    assert r.status_code == 200
    values = {e["value"] for e in r.json()}
    assert values == {
        "western_europe",
        "eastern_europe",
        "west_asia",
        "central_asia",
        "east_asia",
        "southeast_asia",
        "africa",
        "north_america",
        "south_america",
        "oceania",
    }


def test_add_and_delete_ethnicity(client, auth_headers):
    r = client.post("/api/ethnicities", json={"value": "scandinavia", "label": "北欧"}, headers=auth_headers)
    assert r.status_code == 200
    r = client.post("/api/ethnicities", json={"value": "scandinavia", "label": "重复"}, headers=auth_headers)
    assert r.status_code == 400
    r = client.delete("/api/ethnicities/scandinavia", headers=auth_headers)
    assert r.status_code == 200


def test_bases_list_has_seed(client, auth_headers):
    r = client.get("/api/bases", headers=auth_headers)
    assert r.status_code == 200
    ids = {b["id"] for b in r.json()}
    assert "base_women_01" in ids
    women = next(b for b in r.json() if b["id"] == "base_women_01")
    assert women["ethnicity"] == "east_asia"


def test_add_base_and_patch(client, auth_headers):
    r = client.post(
        "/api/bases",
        data={
            "name": "新底座",
            "audience": "women",
            "age_band": "adult",
            "ethnicity": "north_america",
            "notes": "测试",
        },
        files={"file": ("base.png", _png_bytes(), "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 200
    base = r.json()
    assert base["ethnicity"] == "north_america"

    r = client.patch(f"/api/bases/{base['id']}", json={"ethnicity": "east_asia"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["ethnicity"] == "east_asia"

    r = client.patch(f"/api/bases/{base['id']}", json={"ethnicity": "bad_value"}, headers=auth_headers)
    assert r.status_code == 400


def test_add_base_bad_ethnicity_rejected(client, auth_headers):
    r = client.post(
        "/api/bases",
        data={"name": "x", "audience": "women", "ethnicity": "not_real"},
        files={"file": ("b.png", _png_bytes(), "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_batch_patch_bases(client, auth_headers):
    r = client.post(
        "/api/bases/batch/patch",
        json={"ids": ["base_women_01", "base_men_01"], "patch": {"ethnicity": "north_america"}},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_upload_library_and_move(client, auth_headers):
    r = client.post(
        "/api/uploads",
        data={"category": ""},
        files={"file": ("lib.png", _png_bytes((1, 2, 3)), "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 200
    name = r.json()["name"]

    r = client.post(f"/api/uploads/{name}/move", json={"category": "testcat"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["category"] == "testcat"

    cats = client.get("/api/uploads/categories", headers=auth_headers)
    assert cats.status_code == 200
    assert "testcat" in cats.json()


def test_upload_library_rejects_non_image(client, auth_headers):
    r = client.post(
        "/api/uploads",
        data={"category": ""},
        files={"file": ("evil.txt", b"hello", "text/plain")},
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_create_batch_job(client, auth_headers):
    r = client.post(
        "/api/jobs/batch",
        data={
            "base_id": "base_women_01",
            "audience": "auto",
            "garment_type": "auto",
            "output_format": "jpg",
            "background": "white",
        },
        files=[("files", ("dress_a.jpg", _png_bytes((200, 30, 30)), "image/jpeg"))],
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["count"] == 1
    job = r.json()["jobs"][0]
    assert job["status"] == "queued"


def test_batch_job_empty_files_rejected_by_validation(client, auth_headers):
    r = client.post(
        "/api/jobs/batch",
        data={"base_id": "base_women_01", "audience": "auto", "garment_type": "auto"},
        files=[],
        headers=auth_headers,
    )
    assert r.status_code == 422


def test_get_job_not_found(client, auth_headers):
    assert client.get("/api/jobs/job_nope", headers=auth_headers).status_code == 404


def test_push_missing_store_message(client, auth_headers):
    job = create_job_via_api(client, auth_headers)
    assert job is not None
    r = client.post(
        f"/api/jobs/{job['id']}/push",
        json={"store_ids": []},
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_push_needs_done_job(client, auth_headers):
    r = client.post(
        "/api/jobs/batch",
        data={"base_id": "base_women_01", "audience": "women", "garment_type": "upper"},
        files=[("files", ("x.jpg", _png_bytes((20, 40, 60)), "image/jpeg"))],
        headers=auth_headers,
    )
    job = r.json()["jobs"][0]
    r = client.post(
        f"/api/jobs/{job['id']}/push",
        json={"store_ids": ["store_x"]},
        headers=auth_headers,
    )
    assert r.status_code == 400  # not done yet


def create_job_via_api(client, auth_headers):
    seed = client.get("/api/bases", headers=auth_headers).json()
    if not seed:
        return None
    base_id = seed[0]["id"]
    r = client.post(
        "/api/jobs/batch",
        data={"base_id": base_id, "audience": "women", "garment_type": "upper", "output_format": "png", "background": "white"},
        files=[("files", ("ok_img.png", _png_bytes((30, 90, 150)), "image/png"))],
        headers=auth_headers,
    )
    return r.json()["jobs"][0] if r.status_code == 200 else None


def test_stores_crud(client, auth_headers):
    r = client.post(
        "/api/stores",
        json={"name": "旗舰店", "platform": "shopify", "shop_url": "abc.myshopify.com", "api_key": "tok", "product_id": "1"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    store = r.json()
    assert store["embed_url"]

    r = client.patch(f"/api/stores/{store['id']}", json={"name": "改名店"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["name"] == "改名店"

    assert client.delete(f"/api/stores/{store['id']}", headers=auth_headers).status_code == 200


def test_stores_requires_name(client, auth_headers):
    r = client.post("/api/stores", json={"name": "  "}, headers=auth_headers)
    assert r.status_code == 400


def test_download_missing_result(client, auth_headers):
    r = client.post(
        "/api/jobs/batch",
        data={"base_id": "base_women_01", "audience": "women", "garment_type": "upper"},
        files=[("files", ("a.jpg", _png_bytes((5, 5, 5)), "image/jpeg"))],
        headers=auth_headers,
    )
    job = r.json()["jobs"][0]
    assert client.get(f"/api/download/{job['id']}", headers=auth_headers).status_code == 404


def test_files_path_traversal_blocked(client):
    r = client.get("/files/../../etc/passwd")
    assert r.status_code in (404, 400)


def test_update_settings(client, auth_headers):
    r = client.patch("/api/settings", json={"store_name": "MannequinForge Test"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["store_name"] == "MannequinForge Test"


def test_stats(client, auth_headers):
    r = client.get("/api/stats", headers=auth_headers)
    assert r.status_code == 200
    assert "bases" in r.json()