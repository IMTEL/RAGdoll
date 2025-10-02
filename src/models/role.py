from pydantic import BaseModel

class Role(BaseModel):
    name: str
    description: str
    subset_of_corpa: list[int] # index of corpus that the role has access to
