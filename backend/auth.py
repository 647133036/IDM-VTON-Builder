import os
import time
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.getenv("STORE_JWT_SECRET", "idm-vton-store-demo-secret")
JWT_ALG = "HS256"
TOKEN_TTL = 60 * 60 * 24 * 7

USERS = {
    "admin": {
        "password": os.getenv("STORE_ADMIN_PASSWORD", "admin123"),
        "name": "Store Admin",
        "role": "admin",
    },
    "staff": {
        "password": os.getenv("STORE_STAFF_PASSWORD", "staff123"),
        "name": "Merchandiser",
        "role": "staff",
    },
}

bearer = HTTPBearer(auto_error=False)


def issue_token(username: str) -> str:
    user = USERS[username]
    payload = {
        "sub": username,
        "name": user["name"],
        "role": user["role"],
        "exp": int(time.time()) + TOKEN_TTL,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录") from exc


def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="需要登录")
    return decode_token(creds.credentials)
