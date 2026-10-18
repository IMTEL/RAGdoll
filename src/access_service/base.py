from abc import abstractmethod
from datetime import datetime

from src.models.accesskey import AccessKeyCreated


class AbstractAccessService:
    @abstractmethod
    def generate_accesskey(
        self,
        name: str,
        expiry_date: datetime | None,
        agent_id: str,
        view_once: bool = True,
    ) -> AccessKeyCreated:
        """Generates an accesskey and stores its hash, returns the plaintext.

        With ``view_once=False`` the key can be revealed again later.
        """

    @abstractmethod
    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        """Revokes the specified token."""

    @abstractmethod
    def authenticate(self, agent_id: str, access_key: str) -> bool:
        """Authenticates the use of an accesskey."""
