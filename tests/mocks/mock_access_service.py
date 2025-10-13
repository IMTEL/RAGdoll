from datetime import datetime

from src.access_service.base import AbstractAccessService
from src.models.accesskey import AccessKey


class MockAccessService(AbstractAccessService):
    def generate_accesskey(
        self, name: str, expiry_date: datetime, agent_id: str
    ) -> AccessKey:
        return AccessKey(
            key="", name=name, expiry_date=expiry_date, created=datetime.now()
        )

    def authenticate(self, agent_id: str, access_key: str) -> bool:
        return True

    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        return True
