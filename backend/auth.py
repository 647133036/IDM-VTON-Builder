import logging
import os
import secrets
import time
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("atelier.auth")

_env_secret = os.getenv("STORE_JWT_SECRET")
if _env_secret and len(_env_secret) >= 32:
    JWT_SECRET = _env_secret
else:
    JWT_SECRET = secrets.token_urlsafe(48)
    if not _env_secret:
        logger.warning(
            "STORE_JWT_SECRET not set; generated a random secret for this process. "
            "All tokens will be invalidated on restart. Set STORE_JWT_SECRET for production."
        )
    else:
        logger.warning(
            "STORE_JWT_SECRET is shorter than 32 bytes; generated a random secret for this process."
        )

JWT_ALG = "HS256"
TOKEN_TTL = 60 * 60 * 24 * 7


def _resolve_password(env_var: str, fallback_label: str) -> str:
    val = os.getenv(env_var)
    if val:
        return val
    generated = secrets.token_urlsafe(12)
    logger.warning(
        "%s not set; generated a random password for '%s'. "
        "Set %s for production or check process logs for the generated credential.",
        env_var, fallback_label, env_var,
    )
    return generated


USERS = {
    "admin": {
        "password": _resolve_password("STORE_ADMIN_PASSWORD", "admin"),
        "name": "Store Admin",
        "role": "admin",
    },
    "staff": {
        "password": _resolve_password("STORE_STAFF_PASSWORD", "staff"),
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
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Session expired, please log in again") from exc
    sub = payload.get("sub")
    if sub not in USERS:
        raise HTTPException(status_code=401, detail="Session expired, please log in again")
    payload["role"] = USERS[sub]["role"]
    return payload


def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Login required")
    return decode_token(creds.credentials)


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user
