from datetime import datetime


class AccessKey(BaseModel):
    key: str
    name: str | None
    expiery_date: datetime | None