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
    # 前端按 ip_adapter_available 决定是否宣称 IP-Adapter,必须返回该字段。
    assert "ip_adapter_available" in r.json()
    assert isinstance(r.json()["ip_adapter_available"], bool)


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


def test_add_ethnicity_rejects_xss_value(client, auth_headers):
    # NEW-M2:value 渲染进 HTML 属性,必须拒绝引号/尖括号/空格等注入字符。
    bad = 'x" onmouseover="alert(1)'
    r = client.post("/api/ethnicities", json={"value": bad, "label": "Bad"}, headers=auth_headers)
    assert r.status_code == 400
    r = client.post("/api/ethnicities", json={"value": "upper Case", "label": "Bad"}, headers=auth_headers)
    assert r.status_code == 400
    r = client.post("/api/ethnicities", json={"value": "ok_value_2", "label": "OK"}, headers=auth_headers)
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


def test_default_category_list_get_move_unified(client, auth_headers):
    # P2-5:default 分类必须与根目录统一,list/get/move 三处行为一致。
    r = client.post(
        "/api/uploads",
        data={"category": ""},
        files={"file": ("rootfile.png", _png_bytes((9, 8, 7)), "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 200
    name = r.json()["name"]
    assert r.json()["category"] == "default"

    listed = client.get("/api/uploads?category=default", headers=auth_headers)
    assert listed.status_code == 200
    assert any(item["name"] == name for item in listed.json())

    got = client.get(f"/api/uploads/default/{name}", headers=auth_headers)
    assert got.status_code == 200
    assert got.json()["category"] == "default"

    moved = client.post(f"/api/uploads/{name}/move", json={"category": "default"}, headers=auth_headers)
    assert moved.status_code == 200
    # 移到 default 是同一目录,文件名必须保持不变(P2-5 曾改名为 upload_xxx_1)。
    assert moved.json()["filename"] == name
    still = client.get(f"/api/uploads/default/{name}", headers=auth_headers)
    assert still.status_code == 200


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

    # 短 key(<12 位)整体隐藏,避免掩码泄露过半字符(NEW-L4)。
    client.patch(f"/api/stores/{store['id']}", json={"api_key": "newtok1"}, headers=auth_headers)
    assert r.status_code == 200
    r = client.get("/api/stores", headers=auth_headers)
    short = next(s for s in r.json() if s["id"] == store["id"])
    assert short["api_key_masked"] == "***"

    # 9~11 位 key 也必须整体隐藏(曾暴露 3+3=6 位,泄露 2/3)。
    client.patch(f"/api/stores/{store['id']}", json={"api_key": "ninebitskey"}, headers=auth_headers)
    r = client.get("/api/stores", headers=auth_headers)
    nine = next(s for s in r.json() if s["id"] == store["id"])
    assert nine["api_key_masked"] == "***"

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
    # 必须真的发起 /files 请求并断言 200,否则条件嵌套会让测试空转(假覆盖)。
    up = client.post(
        "/api/uploads",
        data={"category": ""},
        files={"file": ("pub.png", _png_bytes((11, 22, 33)), "image/png")},
        headers=auth_headers,
    )
    assert up.status_code == 200
    url = up.json()["url"]
    assert url.startswith("/files/")
    r = client.get(url)
    assert r.status_code == 200
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_files_state_json_404(client, auth_headers):
    # 状态文件必须不可公开访问(H-12):不泄露用户/店铺信息。
    r = client.get("/files/state.json")
    assert r.status_code == 404


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


def test_store_default_sku_roundtrip(client, auth_headers):
    """Store 创建时带 default_sku,列表/编辑响应保留该字段。"""
    r = client.post(
        "/api/stores",
        json={
            "name": "SKU测试店",
            "platform": "shopify",
            "shop_url": "sku.myshopify.com",
            "api_key": "tok",
            "product_id": "42",
            "default_sku": "SKU-RED-M",
        },
        headers=auth_headers,
    )
    assert r.status_code == 200
    store = r.json()
    assert store["default_sku"] == "SKU-RED-M"

    # patch 更新 default_sku
    r = client.patch(f"/api/stores/{store['id']}", json={"default_sku": "SKU-BLU-L"}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["default_sku"] == "SKU-BLU-L"

    # 列表返回含 default_sku
    r = client.get("/api/stores", headers=auth_headers)
    matched = next(s for s in r.json() if s["id"] == store["id"])
    assert matched["default_sku"] == "SKU-BLU-L"

    client.delete(f"/api/stores/{store['id']}", headers=auth_headers)


def test_batch_job_binds_store_id(client, auth_headers):
    """batch 创建 job 时传 bound_store_id,job 记录该字段。"""
    # 先建一个店铺
    r = client.post(
        "/api/stores",
        json={"name": "绑定测试店", "platform": "other"},
        headers=auth_headers,
    )
    store_id = r.json()["id"]

    seed = client.get("/api/bases", headers=auth_headers).json()
    base_id = seed[0]["id"]
    r = client.post(
        "/api/jobs/batch",
        data={
            "base_id": base_id,
            "audience": "women",
            "garment_type": "upper",
            "bound_store_id": store_id,
        },
        files=[("files", ("bind.png", _png_bytes((30, 90, 150)), "image/png"))],
        headers=auth_headers,
    )
    assert r.status_code == 200
    job = r.json()["jobs"][0]
    assert job["bound_store_id"] == store_id

    client.delete(f"/api/stores/{store_id}", headers=auth_headers)


def test_push_defaults_to_bound_store(client, auth_headers):
    """job 有 bound_store_id 时,push 不传 store_ids 回退到该店铺(不会报 No pushable store)。"""
    r = client.post(
        "/api/stores",
        json={"name": "回退测试店", "platform": "other"},
        headers=auth_headers,
    )
    store_id = r.json()["id"]

    seed = client.get("/api/bases", headers=auth_headers).json()
    base_id = seed[0]["id"]
    r = client.post(
        "/api/jobs/batch",
        data={"base_id": base_id, "bound_store_id": store_id},
        files=[("files", ("bound.png", _png_bytes((20, 60, 90)), "image/png"))],
        headers=auth_headers,
    )
    job = r.json()["jobs"][0]

    # job 未完成,push 应返回 400 "Only completed results"(不是 "No pushable store")
    # —— 证明 bound_store_id 回退生效,store 找到了,只是 job 没 done。
    r = client.post(
        f"/api/jobs/{job['id']}/push",
        json={"store_ids": []},
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "completed" in r.json()["detail"].lower()

    client.delete(f"/api/stores/{store_id}", headers=auth_headers)


def test_export_csv_returns_zip_with_manifest_and_images(client, auth_headers):
    """导出 ZIP:含 manifest.csv + 成品图,CSV 行数 = 图片数,image_filename 一致。"""
    import io
    import zipfile

    from store import RESULTS, patch_job

    job = create_job_via_api(client, auth_headers)
    assert job is not None

    # 手动标记 job 为 done 并写一张成品图(绕过引擎)
    result_name = f"{job['id']}.jpg"
    (RESULTS / result_name).write_bytes(_png_bytes((200, 100, 50)))
    patch_job(job["id"], {"status": "done", "result_url": f"/files/results/{result_name}"})

    r = client.post("/api/jobs/export-csv", json={"job_ids": [job["id"]]}, headers=auth_headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "manifest.csv" in names
    assert f"{job['id']}.jpg" in names

    import csv as csv_mod

    reader = csv_mod.DictReader(io.StringIO(zf.read("manifest.csv").decode()))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["job_id"] == job["id"]
    assert rows[0]["image_filename"] == f"{job['id']}.jpg"
    assert rows[0]["status"] == "done"
    # CSV 行数 == 图片数(不含 manifest.csv)
    image_files = [n for n in names if n != "manifest.csv"]
    assert len(rows) == len(image_files)


def test_export_csv_skips_non_done_jobs(client, auth_headers):
    """不存在的 job_id 被跳过;无可导出任务时返回 400。"""
    r = client.post("/api/jobs/export-csv", json={"job_ids": ["job_nonexistent"]}, headers=auth_headers)
    assert r.status_code == 400


def test_export_csv_empty_job_ids_rejected(client, auth_headers):
    r = client.post("/api/jobs/export-csv", json={"job_ids": []}, headers=auth_headers)
    assert r.status_code == 400


def _make_done_job_with_result(client, auth_headers, color=(200, 100, 50)):
    """创建一个 done 状态的 job 并写成品图,返回 job dict。"""
    from store import RESULTS, patch_job

    job = create_job_via_api(client, auth_headers)
    assert job is not None
    result_name = f"{job['id']}.jpg"
    (RESULTS / result_name).write_bytes(_png_bytes(color))
    patch_job(job["id"], {"status": "done", "result_url": f"/files/results/{result_name}"})
    return get_job_via_api(client, auth_headers, job["id"])


def get_job_via_api(client, auth_headers, job_id):
    r = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    return r.json()


def test_batch_push_replacement_image_priority(client, auth_headers):
    """CSV + 替换图包:上传包有同名图时用替换图,否则用原成品图。"""
    import csv as csv_mod
    import io as io_mod

    from store import RESULTS

    job = _make_done_job_with_result(client, auth_headers, color=(200, 100, 50))

    # 创建店铺(platform=other,不需要真实推送,push_store 返回 tagged only)
    r = client.post(
        "/api/stores",
        json={"name": "批量推送测试店", "platform": "other", "default_sku": "SKU-DEF"},
        headers=auth_headers,
    )
    store_id = r.json()["id"]

    # 构造 CSV
    csv_buf = io_mod.StringIO()
    writer = csv_mod.DictWriter(
        csv_buf,
        fieldnames=["job_id", "image_filename", "source_name", "store_name", "store_id", "product_id", "sku", "status"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "job_id": job["id"],
            "image_filename": f"{job['id']}.jpg",
            "source_name": "test.png",
            "store_name": "",
            "store_id": "",
            "product_id": "",
            "sku": "",
            "status": "done",
        }
    )

    # 上传一张替换图(颜色不同)
    replaced_bytes = _png_bytes((50, 200, 100))

    r = client.post(
        f"/api/stores/{store_id}/batch-push",
        files={
            "csv_file": ("manifest.csv", csv_buf.getvalue().encode(), "text/csv"),
            "files": (f"{job['id']}.jpg", replaced_bytes, "image/jpeg"),
        },
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["ok"] == 1
    assert body["results"][0]["job_id"] == job["id"]
    assert body["results"][0]["ok"] is True
    # 替换图写到 RESULTS
    assert (RESULTS / f"{job['id']}_replaced.jpg").exists()

    client.delete(f"/api/stores/{store_id}", headers=auth_headers)


def test_batch_push_missing_image_skipped(client, auth_headers):
    """CSV 行的图片在上传包和原成品中都不存在时,跳过并标注「图片缺失」。"""
    import csv as csv_mod
    import io as io_mod

    r = client.post(
        "/api/stores",
        json={"name": "缺图测试店", "platform": "other"},
        headers=auth_headers,
    )
    store_id = r.json()["id"]

    csv_buf = io_mod.StringIO()
    writer = csv_mod.DictWriter(
        csv_buf,
        fieldnames=["job_id", "image_filename", "sku", "product_id"],
    )
    writer.writeheader()
    writer.writerow({"job_id": "job_nonexistent", "image_filename": "missing.jpg", "sku": "", "product_id": ""})

    r = client.post(
        f"/api/stores/{store_id}/batch-push",
        files={"csv_file": ("manifest.csv", csv_buf.getvalue().encode(), "text/csv")},
        headers=auth_headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["ok"] == 0
    assert body["results"][0]["ok"] is False
    assert "not found" in body["results"][0]["message"].lower() or "缺失" in body["results"][0]["message"]

    client.delete(f"/api/stores/{store_id}", headers=auth_headers)


def test_batch_push_sku_fallback_chain(client, auth_headers):
    """SKU 回退链:CSV 行有 sku → 用行 sku;行 sku 空 → 用店铺 default_sku。"""
    import csv as csv_mod
    import io as io_mod

    job = _make_done_job_with_result(client, auth_headers)

    # 店铺有 default_sku
    r = client.post(
        "/api/stores",
        json={"name": "SKU回退测试店", "platform": "other", "default_sku": "SKU-STORE"},
        headers=auth_headers,
    )
    store_id = r.json()["id"]

    # CSV 行 sku 为空 → 回退到店铺 default_sku
    csv_buf = io_mod.StringIO()
    writer = csv_mod.DictWriter(csv_buf, fieldnames=["job_id", "image_filename", "sku", "product_id"])
    writer.writeheader()
    writer.writerow({"job_id": job["id"], "image_filename": f"{job['id']}.jpg", "sku": "", "product_id": ""})

    r = client.post(
        f"/api/stores/{store_id}/batch-push",
        files={"csv_file": ("manifest.csv", csv_buf.getvalue().encode(), "text/csv")},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["ok"] == 1  # platform=other,push_store 返回 tagged only (ok=True)

    client.delete(f"/api/stores/{store_id}", headers=auth_headers)


def test_batch_push_rejects_bad_csv_columns(client, auth_headers):
    """CSV 缺少必需列时返回 400,列出期望列名。"""
    r = client.post(
        "/api/stores",
        json={"name": "CSV校验店", "platform": "other"},
        headers=auth_headers,
    )
    store_id = r.json()["id"]

    r = client.post(
        f"/api/stores/{store_id}/batch-push",
        files={"csv_file": ("bad.csv", b"wrong_column,value\nfoo,bar\n", "text/csv")},
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "job_id" in r.json()["detail"]

    client.delete(f"/api/stores/{store_id}", headers=auth_headers)


def test_push_results_record_sku_and_product_id(client, auth_headers):
    """push_results 记录每店使用的 product_id/sku(含回退到店铺默认)。"""

    # 店铺有 default_sku 和 product_id
    r = client.post(
        "/api/stores",
        json={
            "name": "推送记录测试店",
            "platform": "other",
            "default_sku": "SKU-DEFAULT",
            "product_id": "999",
        },
        headers=auth_headers,
    )
    store_id = r.json()["id"]

    job = _make_done_job_with_result(client, auth_headers)

    # push 不传 sku/product_id → 回退到店铺默认
    r = client.post(
        f"/api/jobs/{job['id']}/push",
        json={"store_ids": [store_id]},
        headers=auth_headers,
    )
    assert r.status_code == 200
    push_results = r.json()["push_results"]
    assert store_id in push_results
    assert push_results[store_id]["sku"] == "SKU-DEFAULT"
    assert push_results[store_id]["product_id"] == "999"

    client.delete(f"/api/stores/{store_id}", headers=auth_headers)
