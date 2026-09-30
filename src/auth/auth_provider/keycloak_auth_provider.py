import requests
import base64
import logging
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWKClient

from src.auth.auth_provider.base import AuthProvider
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


logger = logging.getLogger(__name__)

@dataclass
class KeycloakUserData:
    id: str
    name: str | None
    email: str | None
    picture: str | None


class KeycloakAuthProvider(AuthProvider):
    def __init__(
        self,
        issuer: str,
        allowed_issuers: list[str] | None,
        jwks_url: str,
        client_id: str,
        verify_audience: bool,
        user_db: UserDao,
    ):
        self.allowed_issuers = {
            issuer.rstrip("/"),
            *(item.rstrip("/") for item in (allowed_issuers or [])),
        }
        self.issuer = issuer.rstrip("/")
        if self.issuer not in self.allowed_issuers:
            raise ValueError("This is not a valid issuer")
        oidc_config = requests.get(f"{issuer}/.well-known/openid-configuration").json()
        self.signing_algos = oidc_config["id_token_signing_alg_values_supported"]
        # This should be fetched trough the oidc_config instead, however docker connection issues is causing some issues with
        # Localhost development testing
        self.jwks_client = PyJWKClient(f"{issuer}/protocol/openid-connect/certs")
        self.jwks_url = jwks_url
        self.client_id = client_id
        self.verify_audience = verify_audience
        self.user_db = user_db
        self._jwks: dict[str, Any] | None = None
        self._jwks_loaded_at = 0.0
        self._jwks_ttl_seconds = 300

    def get_authenticated_user(self, token: str) -> User | None:
        user_data = self.authenticate_user(token)
        user = self.user_db.get_user_by_provider(
            KeycloakAuthProvider.get_provider(), user_data.id
        )
        if user is not None:
            user.name = user_data.name or user.name
            user.email = user_data.email or user.email
            user.picture = user_data.picture or user.picture
            return self.user_db.set_user(user)

        return self.user_db.set_user(
            User(
                auth_provider=KeycloakAuthProvider.get_provider(),
                provider_user_id=user_data.id,
                name=user_data.name,
                email=user_data.email,
                picture=user_data.picture,
                owned_agents=[],
            )
        )

    def authenticate_user(self, token: str) -> KeycloakUserData:
        claims = self._decode_token(token)
        provider_user_id = claims.get("sub")
        if not provider_user_id:
            raise ValueError("Keycloak token is missing subject")

        name = claims.get("name") or claims.get("preferred_username")
        email = claims.get("email")
        picture = claims.get("picture")
        return KeycloakUserData(provider_user_id, name, email, picture)

    def _decode_token(self, token: str) -> dict[str, Any]:
        signing_key = self.jwks_client.get_signing_key_from_jwt(token)
        data = jwt.decode(
            token,
            key=signing_key,
            audience=self.client_id,
            algorithms=self.signing_algos
        )
        return data


    @staticmethod
    def _base64url_decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    @staticmethod
    def get_provider() -> str:
        return "keycloak"
