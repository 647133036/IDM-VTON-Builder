import logging
import os
import secrets
import time
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("atelier.auth")

_SECRET_FILE = Path(__file__).resolve().parent / "data" / ".jwt_secret"


def _load_or_create_jwt_secret() -> str:
    env_val = os.getenv("STORE_JWT_SECRET")
    if env_val and len(env_val) >= 32:
        return env_val
    _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _SECRET_FILE.exists():
        try:
            val = _SECRET_FILE.read_text(encoding="utf-8").strip()
            if len(val) >= 32:
                return val
        except OSError:
            pass
    val = secrets.token_urlsafe(48)
    try:
        _SECRET_FILE.write_text(val, encoding="utf-8")
        os.chmod(_SECRET_FILE, 0o600)
    except OSError as exc:
        logger.warning("Failed to persist JWT secret to %s: %s", _SECRET_FILE, exc)
    if not env_val:
        logger.warning(
            "STORE_JWT_SECRET not set; generated and persisted a random secret to %s. "
            "Set STORE_JWT_SECRET (>=32 bytes) for production.",
            _SECRET_FILE,
        )
    else:
        logger.warning("STORE_JWT_SECRET is shorter than 32 bytes; generated and persisted a random secret instead.")
    return val


JWT_SECRET = _load_or_create_jwt_secret()
JWT_ALG = "HS256"
TOKEN_TTL = 60 * 60 * 24 * 7


def _resolve_password(env_var: str, username: str) -> str:
    val = os.getenv(env_var)
    if val:
        return val
    generated = secrets.token_urlsafe(12)
    print(
        f"[atelier] {env_var} not set; generated random password for '{username}': {generated}",
        flush=True,
    )
    logger.warning("%s not set; generated random password for '%s' (printed to stdout)", env_var, username)
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


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Login required")
    return decode_token(creds.credentials)


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user
