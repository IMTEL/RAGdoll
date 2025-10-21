from pydantic import BaseModel, Field


class User(BaseModel):
    """A user in the authenticated by a spesific authentication provider.

    The user system based on using SSO's from different providers, like Google SSO.
    We have our own id to make sure every user has different id's.


    Attributes:
        id: Unique identifier for the user independent of provider
        authentication_provider: who is resposible for authentication this user
        provider_user_id: the id given by the provider
        name: the name of the user
        email: the email of the user
        picture: picture for use on the config site
        owned_agents the agent_ids of the agents owned by the user
    """

    id: str | None = Field(default=None, description="Unique identifier for the user")
    auth_provider: str
    provider_user_id: str
    name: str | None = None
    email: str | None = None
    picture: str | None = None
    owned_agents: list[str] = []
