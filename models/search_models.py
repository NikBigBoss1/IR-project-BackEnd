from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    filters: Optional[List[str]]