from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from langchain_core.vectorstores import VectorStore
from ...custom_types.graphs.community import CommunityLevel

from .communities_reports import CommunitiesReportsSelector
from .entities import EntitiesSelector
from .relationships import RelationshipsSelectionResult, RelationshipsSelector
from .text_units import TextUnitsSelector

from langchain_community.graphs.neo4j_graph import Neo4jGraph
from dataclasses import dataclass, field

@dataclass
class GraphDataFrame:
    entities: pd.DataFrame
    relationships: pd.DataFrame
    text_units: pd.DataFrame
    communities_reports: pd.DataFrame

@dataclass
class CypherQuery:
    USER_ID: str
    entities_query: str = field(init=False)
    relationships_query: str = field(init=False)
    text_units_query: str = field(init=False)
    communities_reports_query: str = field(init=False)
    
    def __post_init__(self):
    
        self.entities_query:str = field(default=f"""
            MATCH (n) 
            WHERE ANY(label IN labels(n) WHERE label ENDS WITH '_{self.USER_ID}')
            RETURN n.id as id,n.description as description,n.degree as degree,n.text_unit_ids as text_unit_ids,n.communities as communities
        """)
        self.relationships_query:str = field(default=f"""
            MATCH (s:`__Entity__{self.USER_ID}`)-[r]->(t:`__Entity__{self.USER_ID}`)
            RETURN id(r) as id,s.id as source_id,t.id as target_id,r.rank as rank,r.source as source,r.target as target
        """)
        self.text_units_query:str = field(default=f"""
            MATCH (n:`Document__{self.USER_ID}`)-[r]->(m:`__Entity__{self.USER_ID}`)
            RETURN n.id AS id, COLLECT(ID(r)) AS relationship_ids, n.text AS text_unit;
        """)
        self.communities_reports_query:str = field(default=f"""
            MATCH (n:`__Community__{self.USER_ID}`)
            RETURN ID(n) AS id, n.level AS level, n.community_rank AS rating, n.id AS community_id,n.title as title,n.summary as content;
        """)

def getInfoFromNeo4j(graph:Neo4jGraph,USER_ID:str)->GraphDataFrame:
    query = CypherQuery(USER_ID)    
    entites_res = graph.query(query.entities_query.default)
    relationships_res = graph.query(query.relationships_query.default)
    text_units_res = graph.query(query.text_units_query.default)
    communities_reports_res = graph.query(query.communities_reports_query.default)
    # 都不为[]
    assert entites_res != [],"实体记录不存在"
    assert relationships_res != [],"关系记录不存在"
    assert text_units_res != [],"文本单元记录不存在"
    assert communities_reports_res != [],"社区报告记录不存在"    
    entites = pd.DataFrame.from_records(entites_res)
    entites = entites[entites['text_unit_ids'].notna()]
    relationships = pd.DataFrame.from_records(relationships_res)
    text_units = pd.DataFrame.from_records(text_units_res)
    communities_reports = pd.DataFrame.from_records(communities_reports_res)
    return GraphDataFrame(entities=entites,relationships=relationships,text_units=text_units,communities_reports=communities_reports)
    
class ContextSelectionResult(NamedTuple):
    entities: pd.DataFrame
    text_units: pd.DataFrame
    relationships: RelationshipsSelectionResult
    communities_reports: pd.DataFrame

class ContextSelector:
    def __init__(
        self,
        entities_selector: EntitiesSelector,
        text_units_selector: TextUnitsSelector,
        relationships_selector: RelationshipsSelector,
        communities_reports_selector: CommunitiesReportsSelector,
        USER_ID: str,
    ):
        self._entities_selector = entities_selector
        self._text_units_selector = text_units_selector
        self._relationships_selector = relationships_selector
        self._communities_reports_selector = communities_reports_selector
        self._USER_ID = USER_ID

    @staticmethod
    def build_default(
        entities_vector_store: VectorStore,
        entities_top_k: int,
        community_level: CommunityLevel,
        USER_ID: str,
    ) -> ContextSelector:
        
        return ContextSelector(
            entities_selector=EntitiesSelector(
                vector_store=entities_vector_store,
                top_k=entities_top_k,
            ),
            text_units_selector=TextUnitsSelector(),
            relationships_selector=RelationshipsSelector(),
            communities_reports_selector=CommunitiesReportsSelector(
                community_level=community_level
            ),
            USER_ID=USER_ID,
        )

    def run(
        self,
        query: str,
        graph:Neo4jGraph
    ):
        
        # 获取所有实体并转化成df
        # 获取所有关系并转化成df
        # 获取所有文本单元并转化成df
        # 获取所有社区报告并转化成df
        graphDF = getInfoFromNeo4j(graph,self._USER_ID)
        
        # Step 1
        # Select the entities to be used in the local search
        selected_entities = self._entities_selector.run(query, graphDF.entities)

        # Step 2
        # Select the text units to be used in the local search
        selected_text_units = self._text_units_selector.run(
            df_entities=selected_entities,
            df_relationships=graphDF.relationships,
            df_text_units=graphDF.text_units,
        )

        # Step 3
        # Select the relationships to be used in the local search
        selected_relationships = self._relationships_selector.run(
            df_entities=selected_entities,
            df_relationships=graphDF.relationships,
        )

        # Step 4
        # Select the communities to be used in the local search
        selected_communities_reports = self._communities_reports_selector.run(
            df_entities=selected_entities,
            df_reports=graphDF.communities_reports,
        )

        return ContextSelectionResult(
            entities=selected_entities,
            text_units=selected_text_units,
            relationships=selected_relationships,
            communities_reports=selected_communities_reports,
        )