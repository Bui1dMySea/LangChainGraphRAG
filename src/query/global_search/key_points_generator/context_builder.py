import logging

from langchain_core.documents import Document

# from langchain_graphrag.indexing.artifacts import IndexerArtifacts
from ..community_report import CommunityReport
from ..community_weight_calculator import CommunityWeightCalculator
from ...custom_types.graphs.community import CommunityId, CommunityLevel
from ...custom_types.tokens import TokenCounter
from langchain_community.graphs import Neo4jGraph
import pandas as pd 


_REPORT_TEMPLATE = """
--- Report {report_id} ---

Title: {title}
Weight: {weight}
Rank: {rank}
Report:

{content}

"""

_LOGGER = logging.getLogger(__name__)

# 加注释的就说明neo4j里已经有了

class CommunityReportContextBuilder:
    def __init__(
        self,
        community_level: CommunityLevel,
        weight_calculator: CommunityWeightCalculator,
        # artifacts: IndexerArtifacts,
        id:str,
        graph: Neo4jGraph,
        token_counter: TokenCounter,
        max_tokens: int = 8000,
    ):
        self._community_level = community_level
        self._weight_calculator = weight_calculator
        self._graph = graph
        self._id = id
        self._token_counter = token_counter
        self._max_tokens = max_tokens

    def get_df_entities(self) -> pd.DataFrame:
        cypher_query = f"""
            MATCH (e:`__Entity__{self._id}`), (t:`Document{self._id}`)
            WHERE t.text CONTAINS e.id
            WITH e.id AS id, COLLECT(t.id) AS text_unit_ids
            RETURN id, text_unit_ids
        """
        
        # TODO:判断不为空的场景
        
        return pd.DataFrame.from_records(self._graph.query(cypher_query))
        
    # 暂时把content设置成summary
    # TODO:这里的跳数可能需要调整
    def get_df_reports(self):
        cypher_query = f"""
            match (n:`__Community__{self._id}`) 
            where n.summary is not NULL 
            optional match path = (e:`__Entity__{self._id}`)-[*1..3]->(n)
            WHERE ALL(x IN nodes(path) WHERE SINGLE(y IN nodes(path) WHERE y = x))
            RETURN 
                n.id AS community_id,
                n.title AS title,
                n.summary AS summary,
                n.community_rank AS rating,
                n.summary AS content,
                n.level AS level,
                collect(DISTINCT e.id) AS entities
        """
        return pd.DataFrame.from_records(self._graph.query(cypher_query))
    
    def _filter_communities(self) -> list[CommunityReport]:

        df_entities = self.get_df_entities()
        df_reports = self.get_df_reports()
        reports_weight: dict[CommunityId, float] = self._weight_calculator(
            df_entities,
            df_reports,
        )

        df_reports_filtered = df_reports[df_reports["level"] >= self._community_level]
        
        reports = []
        for _, row in df_reports_filtered.iterrows():
            reports.append(
                CommunityReport(
                    id=row["community_id"],
                    weight=reports_weight[row["community_id"]],
                    title=row["title"],
                    summary=row["summary"],
                    rank=row["rating"],
                    content=row["content"],
                )
            )
        return reports

    def __call__(self) -> list[Document]:
        reports = self._filter_communities()
         
        documents: list[Document] = []
        report_str_accumulated: list[str] = []
        token_count = 0
        for report in reports:
            # we would try to combine multiple
            # reports into a single document
            # as long as we do not exceed the token limit
            report_str = _REPORT_TEMPLATE.format(
                report_id=report.id,
                title=report.title,
                weight=report.weight,
                rank=report.rank,
                content=report.content,
            )
            
            report_str_token_count = self._token_counter.count_tokens(report_str)

            if token_count + report_str_token_count > self._max_tokens:
                _LOGGER.warning("Reached max tokens for a community report call ...")
                # we cut a new document here
                documents.append(
                    Document(
                        page_content="\n".join(report_str_accumulated),
                        metadata={"token_count": token_count},
                    )
                )
                # reset the token count and the accumulated string
                token_count = 0
                report_str_accumulated = []
            else:
                token_count += report_str_token_count
                report_str_accumulated.append(report_str)

        if report_str_accumulated:
            documents.append(
                Document(
                    page_content="\n".join(report_str_accumulated),
                    metadata={"token_count": token_count},
                )
            )

        if _LOGGER.isEnabledFor(logging.DEBUG):
            import tableprint

            rows = []
            tableprint.banner("KP Generation Context Token Usage")
            for index, doc in enumerate(documents):
                rows.append([f"Report {index}", doc.metadata["token_count"]])

            tableprint.table(rows, ["Reports", "Token Count"])

        return documents