"""Agent domain models for AI agent configurations."""

from pydantic import BaseModel, Field


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
    access_key: list[str] = Field(
        default_factory=list,
        description="API keys authorized to use this agent",
    )
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

    def is_access_key_valid(self, key: str) -> bool:
        """Verify if the provided access key is authorized for this agent.

        Args:
            key: The access key to validate

        Returns:
            True if the key is authorized, False otherwise
        """
        return not self.access_key or key in self.access_key
