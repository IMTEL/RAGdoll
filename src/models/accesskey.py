from datetime import datetime

from pydantic import BaseModel


class AccessKey(BaseModel):
    """Data model for Access Keys.

    Is used to authorize chatting with roles.
    """

    id: str
    key: str | None
    name: str
    expiry_date: datetime | None
    created: datetime | None
    last_use: datetime | None
