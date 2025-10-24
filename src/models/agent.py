"""Agent domain models for AI agent configurations."""

import logging

from pydantic import BaseModel, Field

from src.models.accesskey import AccessKey
from src.utils.crypto_utils import (
    decrypt_value,
    encrypt_str,
    hash_access_key,
    verify_access_key,
)


logger = logging.getLogger(__name__)


class Role(BaseModel):
    """Role within an agent that defines access to specific document categories.

    A role determines which parts of the agent's knowledge base
    a particular interaction can access by specifying document categories.
    This enables fine-grained control over what information is available for RAG retrieval.

    Attributes:
        name: Unique identifier for the role (e.g., "admin", "user", "viewer")
        description: Human-readable explanation of the role's purpose
        categories: Document categories that this role can access
    """

    name: str = Field(..., description="Unique role identifier")
    description: str = Field(..., description="Role purpose and permissions")
    document_access: list[str] = Field(
        default_factory=list,
        description="Identifiers of documents this role can access",
    )


class Agent(BaseModel):
    """AI Agent configuration for RAG-enabled conversational systems.

    An agent encapsulates all configuration needed to create a specialized
    AI assistant with access to specific knowledge bases and
    role-based access control. Documents are managed through DocumentDAO
    and linked to this agent via agent_id.

    Attributes:
        id: Unique identifier for the agent
        name: Human-readable agent name
        description: Agent's purpose and capabilities
        prompt: System prompt that defines the agent's personality and instructions
        roles: Role definitions for category-based access control
        llm_provider: LLM service provider (e.g., "idun", "openai", "google")
        llm_model: Model identifier (e.g., "gpt-4", "gemini-pro")
        llm_temperature: Sampling temperature (0.0-2.0)
        llm_max_tokens: Maximum tokens in response
        llm_api_key: API key for the LLM service (stored securely in production)
        access_key: List of API keys authorized to use this agent
        retrieval_method: RAG retrieval strategy (e.g., "semantic", "keyword")
        embedding_model: Model for vector embeddings (e.g., "text-embedding-ada-002")
        status: Agent lifecycle status ("active", "inactive", "archived")
        response_format: Expected response format ("text", "json", "markdown")
        last_updated: ISO 8601 timestamp of last modification
    """

    id: str | None = Field(default=None, description="Unique identifier for the agent")
    name: str
    description: str
    prompt: str
    roles: list[Role] = Field(default_factory=list)
    llm_provider: str = Field(default="idun")
    llm_model: str
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1000, gt=0)
    llm_api_key: str = Field(..., description="LLM service API key")
    access_key: list[AccessKey]
    retrieval_method: str = Field(default="semantic")
    embedding_model: str
    status: str = Field(default="active")
    response_format: str = Field(default="text")
    last_updated: str

    def get_role_by_name(self, role_name: str) -> Role | None:
        """Retrieve a role by its name.

        Args:
            role_name: The name of the role to find

        Returns:
            The Role object if found, None otherwise
        """
        for role in self.roles:
            if role.name == role_name:
                return role
        return None

    def get_corpus_for_roles(self, role_names: list[str]) -> list[str]:
        """Get the combined corpus accessible by the given roles.

        Args:
            role_names: List of role names to check

        Returns:
            List of corpus document IDs accessible by any of the roles
        """
        corpus_indices = set()
        for role_name in role_names:
            role = self.get_role_by_name(role_name)
            if role:
                corpus_indices.update(role.subset_of_corpa)

        # Validate indices and return corpus documents
        invalid_indices = [i for i in corpus_indices if i >= len(self.corpa)]
        if invalid_indices:
            if len(self.corpa) == 0:
                valid_range = "none (corpa is empty)"
            else:
                valid_range = f"0-{len(self.corpa) - 1}"
            logger.warning(
                f"Invalid corpus indices for agent '{self.name}': {invalid_indices}. "
                f"Valid range: {valid_range}"
            )

        return [self.corpa[i] for i in corpus_indices if i < len(self.corpa)]

    @classmethod
    def create_with_encryption(
        cls,
        name: str,
        description: str,
        prompt: str,
        llm_model: str,
        embedding_model: str,
        last_updated: str,
        plain_llm_api_key: str,
        plain_access_keys: list[str] | None = None,
        **kwargs,
    ) -> "Agent":
        """Create an Agent with automatic encryption of sensitive data.

        This helper builds an Agent instance, encrypts the provided LLM API key
        and optionally hashes provided access keys before returning the model.
        """
        if not plain_llm_api_key:
            raise ValueError("plain_llm_api_key must not be empty")

        # Create agent with placeholder encrypted key (will be replaced)
        agent = cls(
            name=name,
            description=description,
            prompt=prompt,
            llm_model=llm_model,
            embedding_model=embedding_model,
            last_updated=last_updated,
            llm_api_key="placeholder",
            **kwargs,
        )

        # Set encrypted API key
        agent.set_llm_api_key(plain_llm_api_key)

        # Add hashed access keys
        if plain_access_keys:
            for access_key in plain_access_keys:
                agent.add_access_key(access_key)

        return agent

    def set_llm_api_key(self, plain_api_key: str) -> None:
        """Encrypt and store the LLM API key.

        Args:
            plain_api_key: The plain text API key to encrypt and store
        """
        if not plain_api_key:
            raise ValueError("plain_api_key must not be empty")
        self.llm_api_key = encrypt_str(plain_api_key)

    def get_llm_api_key(self) -> str:
        """Decrypt and return the LLM API key.

        Returns:
            The decrypted API key
        """
        return decrypt_value(self.llm_api_key)

    def add_access_key(self, plain_access_key: str) -> None:
        """Hash and add an access key to the authorized list.

        Args:
            plain_access_key: The plain text access key to hash and store
        """
        if not plain_access_key:
            raise ValueError("plain_access_key must not be empty")
        hashed_key = hash_access_key(plain_access_key)
        # bcrypt hashes are ASCII-safe, so we can store directly as UTF-8 string
        hashed_key_str = hashed_key.decode("utf-8")
        self.access_key.append(hashed_key_str)

    def is_access_key_valid(self, key: str) -> bool:
        """Verify if the provided access key is authorized for this agent.

        Args:
            key: The plain text access key to validate

        Returns:
            True if the key is authorized, False otherwise
        """
        for hashed_key_str in self.access_key:
            try:
                # Convert UTF-8 string back to bytes
                hashed_key = hashed_key_str.encode("utf-8")
                if verify_access_key(key, hashed_key):
                    return True
            except Exception:
                # Skip invalid hash entries
                continue
        return False
