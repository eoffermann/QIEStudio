"""SQLModel table definitions (DESIGN §6).

Importing this package registers every table on ``SQLModel.metadata`` (used by Alembic's
autogenerate and by ``db.create_all`` in tests). Every user-owned table carries a nullable
``owner_id``/``workspace_id`` defaulted to the implicit local owner so multi-user is
additive (DESIGN §4.1a).
"""

from app.models.asset import Asset
from app.models.integration import Integration
from app.models.job import Job, JobInput, JobOutput
from app.models.lora import Lora
from app.models.prompt import Prompt, PromptImage, PromptLora
from app.models.setting import Setting

__all__ = [
    "Asset",
    "Integration",
    "Job",
    "JobInput",
    "JobOutput",
    "Lora",
    "Prompt",
    "PromptImage",
    "PromptLora",
    "Setting",
]
