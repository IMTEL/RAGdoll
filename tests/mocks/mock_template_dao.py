"""Mock implementation of TemplateDAO for testing."""

from itertools import count

from src.models.template import Template
from src.rag_service.dao.template.base import TemplateDAO
from src.utils import singleton


@singleton
class MockTemplateDAO(TemplateDAO):
    """In-memory TemplateDAO. A singleton, so tests can clear the shared state."""

    def __init__(self):
        self.templates: dict[str, Template] = {}
        self._ids = count(1)

    def clear(self) -> None:
        self.templates.clear()

    def create(self, template: Template) -> Template:
        stored = template.model_copy(update={"id": str(next(self._ids))})
        self.templates[stored.id] = stored
        return stored.model_copy()

    def get_by_id(self, template_id: str) -> Template | None:
        template = self.templates.get(template_id)
        return template.model_copy() if template else None

    def get_by_owner(
        self, owner_id: str, template_type: str | None = None
    ) -> list[Template]:
        matches = [
            template.model_copy()
            for template in self.templates.values()
            if template.owner_id == owner_id
            and (template_type is None or template.type == template_type)
        ]
        return sorted(matches, key=lambda template: template.created_at, reverse=True)

    def update(self, template: Template) -> Template:
        if template.id not in self.templates:
            raise ValueError(f"Template with ID {template.id} not found")
        self.templates[template.id] = template.model_copy()
        return template

    def delete(self, template_id: str) -> bool:
        return self.templates.pop(template_id, None) is not None

    def is_reachable(self) -> bool:
        return True
