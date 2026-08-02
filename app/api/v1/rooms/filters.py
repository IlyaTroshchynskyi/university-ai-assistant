from typing import Annotated

from fastapi import Query
from pydantic import BaseModel


# Todo add later
class Pagination(BaseModel):
    cur_page: Annotated[str, Query()]
    next_page: Annotated[str, Query()]
