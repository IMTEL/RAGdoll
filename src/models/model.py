from dataclasses import dataclass


# Lives in the code, does not need basemodel
@dataclass
class Model:
    provider: str
    name: str
    GDPR_compliant: bool | None
    description: str | None
