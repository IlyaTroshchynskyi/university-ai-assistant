from typing import Annotated

from fastapi import Query
from pydantic import BaseModel


# Todo add later
class Pagination(BaseModel):
    current_page: Annotated[str, Query()]
    next_page: Annotated[str, Query()]
