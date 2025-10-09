from pydantic import BaseModel


class Model(BaseModel):
    provider: str
    name: str
    GDPR_compliant: bool | None
    description: str | None
