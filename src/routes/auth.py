from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel

from src.auth.auth_service.factory import auth_service_factory
from src.config import Config
from src.rag_service.dao.factory import get_user_dao


config = Config()
user_dao = get_user_dao()
auth_service = auth_service_factory(config.AUTH_SERVICE)


class Settings(BaseModel):
    authjwt_secret_key: str = Config().JWT_TOKEN_SECRET
    authjwt_denylist_enabled: bool = False
    authjwt_denylist_token_checks: set = {"access", "refresh"}
    authjwt_access_token_expires: timedelta = timedelta(
        minutes=int(config.SESSION_TOKEN_TTL)
    )
    authjwt_refresh_token_expires: timedelta = timedelta(
        days=int(config.REFRESH_TOKEN_TTL)
    )


denylist = set()
router = APIRouter()


# callback to get your configuration
@AuthJWT.load_config
def get_config():
    return Settings()


@router.post("/api/login")
async def login(request: Request, authorize: AuthJWT = Depends()):
    body = await request.json()
    token = body.get("token")
    provider = body.get("provider")
    if not token or not provider:
        raise HTTPException(status_code=400, detail="Missing token or provider")
    try:
        user_id = auth_service.login_user(token, provider)
    except Exception:
        raise HTTPException(status_code=401, detail="Failed to login user")  # noqa: B904

    session_token = authorize.create_access_token(subject=user_id)
    refresh_token = authorize.create_refresh_token(subject=user_id)
    return {
        "session_token": session_token,
        "refresh_token": refresh_token,
        "session_token_ttl": int(config.SESSION_TOKEN_TTL) * 1000 * 60,  # Minutes
        "refresh_token_ttl": int(config.REFRESH_TOKEN_TTL)
        * 1000
        * 60
        * 60
        * 24,  # Days
    }


@router.post("/api/refresh")
def refresh(authorize: AuthJWT = Depends()):
    authorize.jwt_refresh_token_required()
    user_id = authorize.get_jwt_subject()
    new_session_token = authorize.create_access_token(subject=user_id)
    return {
        "session_token": new_session_token,
        "session_token_ttl": config.SESSION_TOKEN_TTL,
    }


# Todo : add to blocking databse


@router.delete("/api/logout")
def logout(authorize: AuthJWT = Depends()):
    authorize.jwt_required()
    jti = authorize.get_raw_jwt()["jti"]
    denylist.add(jti)
    return {"detail": "Tokens has been revolked"}
