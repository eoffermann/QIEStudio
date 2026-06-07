"""Request/response schemas for the catalog/batch export API (DESIGN §13 #5, §5.6).

The export itself streams as a ``application/zip`` (see :mod:`app.routers.export`); these
models only describe the optional explicit-``job_ids`` selection body used when exporting an
ad-hoc set of jobs instead of a whole batch.
"""

from __future__ import annotations

from pydantic import BaseModel


class ExportSelection(BaseModel):
    """Body for exporting an explicit set of jobs (DESIGN §5.6, §13 #5).

    When exporting a batch the batch id is taken from the path instead; this body covers
    the alternate "export these specific jobs" selection.
    """

    job_ids: list[str]
