from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    filters: Optional[List[str]] = None
    cluster_id: Optional[int] = None

class ClusterRequest(BaseModel):
    text_col: str
    num_clusters: int
    num_keywords: Optional[int] = 5