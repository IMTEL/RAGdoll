from dataclasses import dataclass

from src.access_service.access_service import AccessService
from src.rag_service.dao.agent.base import AgentDAO
from tests.mocks.mock_access_service import MockAccessService


@dataclass
class AccessServiceConfig:
    type: str
    database: AgentDAO | None


def access_service_factory(config: AccessServiceConfig):
    match config.type.lower():
        case "mock":
            return MockAccessService()
        case "service":
            return AccessService(config.database)
        case _:
            raise ValueError(f"AccessService {config.type} does not exist")
