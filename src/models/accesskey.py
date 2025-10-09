from datetime import datetime

from pydantic import BaseModel


class AccessKey(BaseModel):
    id: str
    key: str
    name: str
    expiery_date: datetime | None
    created: datetime | None
    last_use: datetime | None
