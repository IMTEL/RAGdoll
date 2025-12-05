import logging
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel

from src.config import Config
from src.globals import auth_service, user_dao


config = Config()
logger = logging.getLogger(__name__)


class Settings(BaseModel):
    authjwt_secret_key: str = Config().JWT_TOKEN_SECRET
    authjwt_denylist_enabled: bool = True
    authjwt_denylist_token_checks: set = {"access", "refresh"}
    authjwt_access_token_expires: timedelta = timedelta(
        minutes=int(config.SESSION_TOKEN_TTL)
    )
    authjwt_refresh_token_expires: timedelta = timedelta(
        days=int(config.REFRESH_TOKEN_TTL)
    )


router = APIRouter()


# callback to get your configuration
@AuthJWT.load_config
def get_config():
    return Settings()


@router.post("/api/login")
async def login(request: Request, authorize: Annotated[AuthJWT, Depends()] = None):
    # If authentication is disabled, return mock tokens and default user
    if config.DISABLE_AUTH:
        logger.info("Login bypassed - authentication disabled")
        user = auth_service._get_or_create_default_user()
        return {
            "session_token": "disabled_auth_token",
            "refresh_token": "disabled_auth_refresh",
            "session_token_ttl": int(config.SESSION_TOKEN_TTL) * 1000 * 60,
            "refresh_token_ttl": int(config.REFRESH_TOKEN_TTL) * 1000 * 60 * 60 * 24,
            "name": user.name,
            "picture": user.picture,
        }
    
    body = await request.json()
    token = body.get("token")
    provider = body.get("provider")
    if not token or not provider:
        raise HTTPException(status_code=400, detail="Missing token or provider")
    try:
        user_id = auth_service.login_user(token, provider)
    except Exception as e:
        logger.warning(f"Failed to login user : {e}")
        raise HTTPException(status_code=401, detail="Failed to login user")  # noqa: B904

    session_token = authorize.create_access_token(subject=user_id)
    refresh_token = authorize.create_refresh_token(subject=user_id)

    user = user_dao.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="To user found")

    return {
        "session_token": session_token,
        "refresh_token": refresh_token,
        "session_token_ttl": int(config.SESSION_TOKEN_TTL) * 1000 * 60,  # Minutes
        "refresh_token_ttl": int(config.REFRESH_TOKEN_TTL) * 1000 * 60 * 60 * 24,
        # Days
        # User data
        "name": user.name,
        "picture": user.picture,
    }


@router.post("/api/refresh")
def refresh(authorize: Annotated[AuthJWT, Depends()] = None):
    # If authentication is disabled, return mock token
    if config.DISABLE_AUTH:
        logger.info("🔓 Token refresh bypassed - authentication disabled")
        return {
            "session_token": "disabled_auth_token",
            "session_token_ttl": config.SESSION_TOKEN_TTL,
        }
    
    authorize.jwt_refresh_token_required()
    user_id = authorize.get_jwt_subject()
    new_session_token = authorize.create_access_token(subject=user_id)
    return {
        "session_token": new_session_token,
        "session_token_ttl": config.SESSION_TOKEN_TTL,
    }


# TODO : imlpement redis database, to avoid filling memory described in https://indominusbyte.github.io/fastapi-jwt-auth/usage/revoking/
denylist = set()


@router.get("/api/logout")
def logout(authorize: Annotated[AuthJWT, Depends()] = None):
    # If authentication is disabled, just return success
    if config.DISABLE_AUTH:
        logger.info("🔓 Logout bypassed - authentication disabled")
        return {"detail": "Logout successful (auth disabled)"}
    
    authorize.jwt_required()
    jti = authorize.get_raw_jwt()["jti"]
    denylist.add(jti)
    return {"detail": "Tokens has been revolked"}


@router.get("/api/auth-status")
def auth_status():
    """Get current authentication status."""
    return {
        "auth_enabled": not config.DISABLE_AUTH,
        "auth_disabled": config.DISABLE_AUTH,
        "message": "Authentication is disabled - using default user" if config.DISABLE_AUTH else "Authentication is enabled"
    }


@AuthJWT.token_in_denylist_loader
def check_if_token_in_denylist(decrypted_token):
    jti = decrypted_token["jti"]
    return jti in denylist
