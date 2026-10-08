from abc import ABC, abstractmethod

from src.models.template import Template


class TemplateDAO(ABC):
    @abstractmethod
    def create(self, template: Template) -> Template:
        """Store a new template and return it with its id."""

    @abstractmethod
    def get_by_id(self, template_id: str) -> Template | None:
        """Retrieve a template by id, or None if there is none."""

    @abstractmethod
    def get_by_owner(
        self, owner_id: str, template_type: str | None = None
    ) -> list[Template]:
        """Retrieve a user's templates, newest first, optionally of one type only."""

    @abstractmethod
    def update(self, template: Template) -> Template:
        """Replace a stored template. Raises ValueError if it does not exist."""

    @abstractmethod
    def delete(self, template_id: str) -> bool:
        """Delete a template. Returns False if there was none with that id."""

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible."""
