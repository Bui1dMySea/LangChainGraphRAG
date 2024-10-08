from typing import List, Optional
from pydantic import BaseModel,Field


class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )
    
class GetTitle(BaseModel):
    title: str = Field(description="Title of the given summary")