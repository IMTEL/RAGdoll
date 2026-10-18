from datetime import UTC, datetime

from src.access_service.base import AbstractAccessService
from src.models.accesskey import AccessKeyCreated


class MockAccessService(AbstractAccessService):
    def generate_accesskey(
        self, name: str, expiry_date: datetime | None, agent_id: str
    ) -> AccessKeyCreated:
        return AccessKeyCreated(
            id="mockkey0",
            key="rgd_mockkey0_mock-secret",
            prefix="rgd_mockkey0",
            name=name,
            expiry_date=expiry_date,
            created=datetime.now(UTC),
        )

    def authenticate(self, agent_id: str, access_key: str) -> bool:
        return True

    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        return True
