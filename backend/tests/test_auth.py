import importlib
import time

import auth
import jwt
import pytest
from auth import JWT_ALG, JWT_SECRET, decode_token, issue_token
from fastapi import HTTPException


def test_issue_and_decode_roundtrip():
    token = issue_token("admin")
    payload = decode_token(token)
    assert payload["sub"] == "admin"
    assert payload["role"] == "admin"
    assert "exp" in payload


def test_decode_staff_role():
    token = issue_token("staff")
    payload = decode_token(token)
    assert payload["role"] == "staff"
    assert payload["name"] == "Merchandiser"


def test_decode_wrong_secret_raises():
    token = jwt.encode({"sub": "admin", "exp": int(time.time()) + 100}, "wrong-secret", algorithm=JWT_ALG)
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_decode_expired_token_raises():
    token = jwt.encode({"sub": "admin", "exp": int(time.time()) - 10}, JWT_SECRET, algorithm=JWT_ALG)
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_decode_malformed_raises():
    with pytest.raises(HTTPException):
        decode_token("not.a.jwt")


def test_unknown_user_issue_raises():
    with pytest.raises(KeyError):
        issue_token("nobody")


def test_short_jwt_env_fails_closed(monkeypatch, tmp_path):
    # 短 env 密钥必须 fail closed,而不是静默降级/复用旧文件(NEW-M9)。
    monkeypatch.setenv("STORE_JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="STORE_JWT_SECRET is only 5 bytes"):
        importlib.reload(auth)


def test_password_shared_across_workers(monkeypatch, tmp_path):
    # 未设置密码时:两次"进程"导入必须拿到同一密码(多 worker 一致, NEW-H2)。
    import auth

    monkeypatch.delenv("STORE_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("STORE_STAFF_PASSWORD", raising=False)
    monkeypatch.setattr(
        auth,
        "_PASSWORD_FILES",
        {
            "admin": tmp_path / ".admin_password",
            "staff": tmp_path / ".staff_password",
        },
    )
    p1 = auth._resolve_password("STORE_ADMIN_PASSWORD", "admin")
    p2 = auth._resolve_password("STORE_ADMIN_PASSWORD", "admin")
    assert p1 == p2
    f = tmp_path / ".admin_password"
    assert f.exists()
    assert f.stat().st_mode & 0o777 == 0o600
    assert f.read_text(encoding="utf-8").strip() == p1


def test_password_env_takes_precedence(monkeypatch, tmp_path):
    import auth

    monkeypatch.setenv("STORE_ADMIN_PASSWORD", "explicit-value")
    monkeypatch.setattr(
        auth,
        "_PASSWORD_FILES",
        {
            "admin": tmp_path / ".admin_password",
            "staff": tmp_path / ".staff_password",
        },
    )
    assert auth._resolve_password("STORE_ADMIN_PASSWORD", "admin") == "explicit-value"
    assert not (tmp_path / ".admin_password").exists()
