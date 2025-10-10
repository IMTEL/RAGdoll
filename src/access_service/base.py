from abc import abstractmethod
from datetime import datetime

from src.models.accesskey import AccessKey


class AbstractAccessService:
    @abstractmethod
    def generate_accesskey(
        self, name: str, expiry_date: datetime, agent_id: str
    ) -> AccessKey:
        """Generates an accesskey and adds it into the database, returns the generated key."""

    @abstractmethod
    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        """Revokes the specified token."""

    @abstractmethod
    def authenticate(self, agent_id: str, access_key: str) -> bool:
        """Authenticates the use of an accesskey."""
