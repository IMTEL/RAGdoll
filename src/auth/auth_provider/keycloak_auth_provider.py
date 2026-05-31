import base64
import logging
from dataclasses import dataclass
from time import monotonic
from typing import Any

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric import rsa

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
        jwks_url: str,
        client_id: str,
        verify_audience: bool,
        user_db: UserDao,
    ):
        self.issuer = issuer.rstrip("/")
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
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            raise ValueError("Keycloak token is missing key id")

        public_key = self._get_public_key(kid)
        decode_kwargs: dict[str, Any] = {
            "key": public_key,
            "algorithms": ["RS256"],
            "issuer": self.issuer,
        }
        if self.verify_audience:
            decode_kwargs["audience"] = self.client_id
        else:
            decode_kwargs["options"] = {"verify_aud": False}

        claims = jwt.decode(token, **decode_kwargs)
        if claims.get("azp") and claims["azp"] != self.client_id:
            logger.debug(
                "Keycloak token authorized party '%s' does not match configured client '%s'",
                claims["azp"],
                self.client_id,
            )
        return claims

    def _get_public_key(self, kid: str):
        jwks = self._get_jwks()
        key = next((item for item in jwks.get("keys", []) if item.get("kid") == kid), None)
        if key is None:
            self._jwks = None
            jwks = self._get_jwks()
            key = next(
                (item for item in jwks.get("keys", []) if item.get("kid") == kid),
                None,
            )
        if key is None:
            raise ValueError("No matching Keycloak signing key found")
        if key.get("kty") != "RSA":
            raise ValueError("Unsupported Keycloak signing key type")

        n = int.from_bytes(self._base64url_decode(key["n"]), byteorder="big")
        e = int.from_bytes(self._base64url_decode(key["e"]), byteorder="big")
        return rsa.RSAPublicNumbers(e, n).public_key()

    def _get_jwks(self) -> dict[str, Any]:
        now = monotonic()
        if self._jwks and now - self._jwks_loaded_at < self._jwks_ttl_seconds:
            return self._jwks

        response = requests.get(self.jwks_url, timeout=10)
        response.raise_for_status()
        self._jwks = response.json()
        self._jwks_loaded_at = now
        return self._jwks

    @staticmethod
    def _base64url_decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    @staticmethod
    def get_provider() -> str:
        return "keycloak"
