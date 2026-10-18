"""Agent domain models for AI agent configurations."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.models.accesskey import AccessKey, AccessKeyPublic


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
    function_access: list[str] = Field(
        default_factory=list,
        description="Function names this role is allowed to call",
    )


class AgentFunction(BaseModel):
    """Function/tool configuration available to an external application."""

    name: str = Field(..., description="External function name")
    required_fields: list[str | dict[str, str]] = Field(
        default_factory=list,
        description="JSON argument fields the LLM must provide",
    )
    call_instructions: str = Field(
        default="",
        description="Instructions for when this function should be called",
    )
    explanation: str | None = Field(
        default=None,
        description="Optional user-facing explanation of the function",
    )
    example_output: str | None = Field(
        default=None,
        description="Optional JSON example output for this function",
    )


class AgentBase(BaseModel):
    """Agent fields without secrets, shared by ``Agent`` and ``AgentPublic``."""

    id: str | None = Field(default=None, description="Unique identifier for the agent")
    name: str
    description: str
    prompt: str
    roles: list[Role] = Field(default_factory=list)
    functions: list[AgentFunction] = Field(default_factory=list)
    llm_provider: str = Field(default="idun")
    llm_model: str
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1000, gt=0)
    retrieval_method: str = Field(default="semantic")
    embedding_model: str
    status: str = Field(default="active")
    response_format: str = Field(default="text")
    last_updated: str

    # RAG retrieval configuration
    top_k: int = Field(
        default=5, gt=0, description="Number of context chunks to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for context relevance",
    )
    hybrid_search_alpha: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Weight for hybrid search (0=keyword only, 1=vector only)",
    )

    context_aware_retrieval: bool = Field(
        default=True,
        description="If true, use context-aware retrieval; otherwise, use only the latest message as the retrieval query.",
    )

    on_retrieval_error: Literal["fail", "answer_without_context"] = Field(
        default="fail",
        description=(
            "What to do when knowledge retrieval fails: 'fail' ends the request "
            "with HTTP 503; 'answer_without_context' answers but tells the "
            "character it has no reference material."
        ),
    )

    # Optional reference to one of the agent owner's stored provider keys
    # (UserAPIKey.id), resolved server-side when the key is needed. Not secret.
    llm_api_key_id: str | None = None
    embedding_api_key_id: str | None = None

    # Set when loading or resolving, never stored: the provider key could not
    # be decrypted (FERNET_KEY changed or was rotated out) or the referenced
    # stored key no longer exists, so it must be entered again.
    llm_api_key_invalid: bool = False
    embedding_api_key_invalid: bool = False


class Agent(AgentBase):
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

    llm_api_key: str = Field(..., description="LLM service API key")
    access_key: list[AccessKey] = Field(default_factory=list)
    embedding_api_key: str = Field(..., description="Embedding service API key")

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


class AgentPublic(AgentBase):
    """Agent as returned by the API: never contains a secret.

    Provider API keys are reduced to ``has_*`` flags, and access keys to their
    metadata (``AccessKeyPublic``).
    """

    access_key: list[AccessKeyPublic] = Field(default_factory=list)
    has_llm_api_key: bool = False
    has_embedding_api_key: bool = False

    @classmethod
    def from_agent(
        cls, agent: Agent, include_access_keys: bool = True
    ) -> "AgentPublic":
        """Builds the public view of ``agent``.

        Args:
            agent: The stored agent
            include_access_keys: False hides the access-key list entirely (for
                callers that authenticated with an access key)
        """
        fields = agent.model_dump(
            exclude={"llm_api_key", "embedding_api_key", "access_key"}
        )
        access_keys = (
            [AccessKeyPublic.model_validate(k.model_dump()) for k in agent.access_key]
            if include_access_keys
            else []
        )
        return cls(
            **fields,
            access_key=access_keys,
            has_llm_api_key=bool(agent.llm_api_key or agent.llm_api_key_id),
            has_embedding_api_key=bool(
                agent.embedding_api_key or agent.embedding_api_key_id
            ),
        )
