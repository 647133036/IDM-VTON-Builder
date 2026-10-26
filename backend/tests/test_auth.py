import time

import jwt
import pytest
from fastapi import HTTPException

from auth import JWT_ALG, JWT_SECRET, decode_token, issue_token


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