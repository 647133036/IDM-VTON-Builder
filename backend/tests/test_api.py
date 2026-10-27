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
        data={
            "base_id": base_id,
            "audience": "women",
            "garment_type": "upper",
            "output_format": "png",
            "background": "white",
        },
        files=[("files", ("ok_img.png", _png_bytes((30, 90, 150)), "image/png"))],
        headers=auth_headers,
    )
    return r.json()["jobs"][0] if r.status_code == 200 else None


def test_stores_crud(client, auth_headers):
    r = client.post(
        "/api/stores",
        json={
            "name": "旗舰店",
            "platform": "shopify",
            "shop_url": "abc.myshopify.com",
            "api_key": "tok",
            "product_id": "1",
        },
        headers=auth_headers,
    )
    assert r.status_code == 200
    store = r.json()
    assert store["embed_url"]
    assert store["api_key"] == ""
    assert store["has_api_key"] is True

    r = client.patch(f"/api/stores/{store['id']}", json={"name": "改名店"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["name"] == "改名店"
    assert r.json()["api_key"] == ""

    r = client.patch(
        f"/api/stores/{store['id']}",
        json={"api_key": "shpat_newtok_12345"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["api_key"] == ""
    assert r.json()["has_api_key"] is True
    # 长 key 仅暴露首尾各 3 位。
    assert r.json()["api_key_masked"] == "shp***345"
    assert "shpat_newtok_12345" not in r.text

    # 短 key(<=8 位)整体隐藏,避免掩码泄露过半字符。
    client.patch(f"/api/stores/{store['id']}", json={"api_key": "newtok1"}, headers=auth_headers)
    assert r.status_code == 200
    r = client.get("/api/stores", headers=auth_headers)
    short = next(s for s in r.json() if s["id"] == store["id"])
    assert short["api_key_masked"] == "***"

    assert client.delete(f"/api/stores/{store['id']}", headers=auth_headers).status_code == 200


def test_download_missing_job(client, auth_headers):
    r = client.get("/api/download/job_does_not_exist", headers=auth_headers)
    assert r.status_code == 404


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
    assert client.get("/files/../../etc/passwd").status_code == 404


def test_files_state_json_blocked(client, auth_headers):
    assert client.get("/files/state.json", headers=auth_headers).status_code == 404
    assert client.get("/files/state.json").status_code == 404


def test_files_public_image_no_auth(client, auth_headers):
    r = client.get("/api/bases", headers=auth_headers)
    bases = r.json()
    if bases:
        url = bases[0].get("url", "")
        if url:
            path = url.replace("/files/", "")
            assert client.get(f"/files/{path}").status_code == 200


def test_upload_html_rejected(client, auth_headers):
    r = client.post(
        "/api/uploads",
        files=[("file", ("evil.html", b"<script>alert(1)</script>", "image/png"))],
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_upload_category_traversal_blocked(client, auth_headers):
    r = client.post(
        "/api/uploads/categories",
        json={"name": ".."},
        headers=auth_headers,
    )
    assert r.status_code == 400
    r2 = client.post(
        "/api/uploads/categories",
        json={"name": "."},
        headers=auth_headers,
    )
    assert r2.status_code == 400


def test_delete_upload_dot_blocked(client, auth_headers):
    r = client.delete("/api/uploads/%2E", headers=auth_headers)
    assert r.status_code == 400


def test_stores_api_key_masked(client, auth_headers):
    r = client.post(
        "/api/stores",
        json={"name": "Test Store", "platform": "shopify", "api_key": "shpat_super_secret_token_12345"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    store_id = r.json()["id"]
    listing = client.get("/api/stores", headers=auth_headers)
    found = next(s for s in listing.json() if s["id"] == store_id)
    assert found["api_key"] == ""
    assert found["has_api_key"] is True
    assert "super_secret" not in found.get("api_key_masked", "")


def test_staff_cannot_delete_base(client):
    r = client.post(
        "/api/auth/login",
        json={"username": "staff", "password": "staff123"},
    )
    assert r.status_code == 200
    staff_headers = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.delete("/api/bases/base_women_01", headers=staff_headers).status_code == 403


def test_staff_cannot_push(client, auth_headers):
    r = client.post(
        "/api/auth/login",
        json={"username": "staff", "password": "staff123"},
    )
    staff_headers = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.post(
        "/api/jobs/base_women_01/push",
        json={"sku": "test"},
        headers=staff_headers,
    ).status_code in (403, 404)


def test_staff_cannot_manage_bases_categories_or_moves(client):
    """H-02: staff 不能新建/改底座、批量改底座、移动文件、建分类(仅 admin)。"""
    r = client.post(
        "/api/auth/login",
        json={"username": "staff", "password": "staff123"},
    )
    assert r.status_code == 200
    staff_headers = {"Authorization": f"Bearer {r.json()['token']}"}

    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 60, 60)).save(buf, format="PNG")
    png = buf.getvalue()

    r = client.post(
        "/api/bases",
        data={"name": "staff 底座", "audience": "women", "ethnicity": "east_asia"},
        files={"file": ("b.png", png, "image/png")},
        headers=staff_headers,
    )
    assert r.status_code == 403
    assert (
        client.post(
            "/api/bases/batch/patch",
            json={"ids": ["base_women_01"], "patch": {"ethnicity": "north_america"}},
            headers=staff_headers,
        ).status_code
        == 403
    )
    assert (
        client.patch("/api/bases/base_women_01", json={"ethnicity": "north_america"}, headers=staff_headers).status_code
        == 403
    )

    # 先由 admin 上传一个文件,再验证 staff 不能移动/建分类。
    admin = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    admin_headers = {"Authorization": f"Bearer {admin.json()['token']}"}
    up = client.post(
        "/api/uploads",
        data={"category": ""},
        files={"file": ("move_me.png", png, "image/png")},
        headers=admin_headers,
    )
    assert up.status_code == 200
    filename = up.json()["name"]

    assert (
        client.post(f"/api/uploads/{filename}/move", json={"category": "staffcat"}, headers=staff_headers).status_code
        == 403
    )
    assert client.post("/api/uploads/categories", json={"name": "staffcat"}, headers=staff_headers).status_code == 403


def test_update_settings(client, auth_headers):
    r = client.patch("/api/settings", json={"store_name": "MannequinForge Test"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["store_name"] == "MannequinForge Test"


def test_stats(client, auth_headers):
    r = client.get("/api/stats", headers=auth_headers)
    assert r.status_code == 200
    assert "bases" in r.json()


def _gif_bytes():
    import io

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, "GIF")
    return buf.getvalue()


def test_upload_rejects_gif_format(client, auth_headers):
    r = client.post(
        "/api/uploads",
        files={"file": ("anim.gif", _gif_bytes(), "image/gif")},
        data={"category": "tops"},
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "GIF" in r.json()["detail"]


def test_settings_output_long_side_range_enforced(client, auth_headers):
    r = client.patch("/api/settings", json={"output_long_side": 64}, headers=auth_headers)
    assert r.status_code == 422
    r = client.patch("/api/settings", json={"output_long_side": 8192}, headers=auth_headers)
    assert r.status_code == 422


def test_settings_background_literal_enforced(client, auth_headers):
    r = client.patch("/api/settings", json={"background": "black"}, headers=auth_headers)
    assert r.status_code == 422


def test_batch_invalid_audience_rejected(client, auth_headers):
    r = client.post(
        "/api/jobs/batch",
        files={"files": ("garment.png", _png_bytes(), "image/png")},
        data={
            "base_id": "base_women_01",
            "audience": "aliens",
            "garment_type": "upper",
            "output_format": "jpg",
            "background": "white",
        },
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "audience" in r.json()["detail"]


def test_batch_invalid_background_rejected(client, auth_headers):
    r = client.post(
        "/api/jobs/batch",
        files={"files": ("garment.png", _png_bytes(), "image/png")},
        data={
            "base_id": "base_women_01",
            "audience": "women",
            "garment_type": "upper",
            "output_format": "jpg",
            "background": "black",
        },
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "background" in r.json()["detail"]


def test_pixel_bomb_rejected(client, auth_headers):
    """H-10: 超大像素图(解压炸弹)必须 400,即使字节数很小。"""
    import io

    from PIL import Image

    # 6000x6000 = 36M 像素,PNG 压缩后远小于 20MB。
    buf = io.BytesIO()
    Image.new("1", (9000, 9000), 0).save(buf, format="PNG")
    r = client.post(
        "/api/uploads",
        data={"category": ""},
        files={"file": ("bomb.png", buf.getvalue(), "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "pixels" in r.json()["detail"]


def test_batch_count_limit(client, auth_headers):
    """H-10: 批量超过 _MAX_BATCH_FILES 张必须 400。"""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(buf, format="PNG")
    png = buf.getvalue()
    files = [("files", (f"g{i}.png", png, "image/png")) for i in range(51)]
    r = client.post(
        "/api/jobs/batch",
        data={"base_id": "base_women_01", "audience": "auto", "garment_type": "auto"},
        files=files,
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "Too many files" in r.json()["detail"]


def test_move_to_default_does_not_create_subdir(client, auth_headers):
    """P2-5: move/upload 到保留字 default 应落在根目录,不建 default 子目录。"""
    r = client.post(
        "/api/uploads",
        data={"category": "default"},
        files={"file": ("d.png", _png_bytes((9, 9, 9)), "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["category"] == "default"


def test_health_does_not_leak_abs_paths(client):
    """P2-11: /api/health 不得暴露本地绝对路径。"""
    r = client.get("/api/health")
    assert r.status_code == 200
    detail = r.json()["engine_detail"]
    assert "weights_dir" not in detail
    assert "idm_vton_dir" not in detail
    assert "sdxl_dir" not in detail
    assert "/" not in str(detail.get("idm_vton_reason", "")) or "no" in str(detail.get("idm_vton_reason", "")).lower()
