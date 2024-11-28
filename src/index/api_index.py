import os
import pandas as pd
from typing import  Dict, List
from tqdm.asyncio import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI,OpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters.base import TextSplitter
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents import Document
# Graph
import json_repair
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
# utils
from .utils import num_tokens_from_string,create_prompt,process_text,entity_resolution,process_communities,process_summaries,countNodesMerged
from ..utils.logger import create_rotating_logger
from logging import Logger
from graphdatascience import GraphDataScience
from .cypher_query import CypherQuery

# prompt
from .prompts import SystemPrompts, UserPrompts
# hf
from transformers import AutoTokenizer
# pydantic models
from .pydantic_models import Disambiguate, GetTitle

COMMUNITY_TEMPLEATE = """Based on the provided nodes and relationships that belong to the same graph community,
        generate a natural language summary of the provided information:
        {community_info}

        Summary:"""  # noqa: E501

community_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input triples, generate the information summary. No pre-amble.",
        ),
        ("human", COMMUNITY_TEMPLEATE),
    ]
)

TITLE_TEMPLATE = """Given the following summary, provide a title that best represents the content:
        {summary}
        
        Title:"""
        
title_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a summary, generate a title that best represents the content. No pre-amble.",
        ),
        ("human", TITLE_TEMPLATE),
    ]
)


class ApiIndex(object):
    def __init__(
        self,
        graph:Neo4jGraph,
        chat_model:BaseChatModel,
        embedding:Embeddings,
        splitter:TextSplitter,
        gds:GraphDataScience,
        logger:Logger=None,
        max_workers:int=4,
        gds_similarity_threshold:float=0.95,
        word_edit_distance:int = 3,
        uuid:str="",
        model_name="gpt-4o-mini"
    ):
        self.graph = graph
        self.chat_model = chat_model
        self.model_name = model_name        
        self.embedding = embedding
        self.splitter = splitter
        self.gds = gds
        self.cypherQuery = CypherQuery(graph=graph)
        if not logger:
            self.logger = create_rotating_logger("index")
        else:
            self.logger = logger
        
        self.MAX_WORKERS = max_workers
        self.GDS_SIMILARITY_THRESHOLD = gds_similarity_threshold
        self.WORD_EDIT_DISTANCE = word_edit_distance
        self.uuid = uuid
        
    def _preprocess(self,documents:List[Dict[str,str]]):
        self.logger.info("Chunking documents")
        data = []
        for document in documents:
            title,text = document['title'],document['text']
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                data.append({"title": title, "text": chunk})

        return pd.DataFrame(data)
    
    def _parse_hf_ollama(self,content:str,source:Document):
        try:
            breakpoint()
            parsed_json = json_repair.loads(content)
            relationships = []
            nodes_set = set()
            for rel in parsed_json:
                # Nodes need to be deduplicated using a set
                if "head_description" in rel.keys():
                    nodes_set.add((rel["head"], rel["head_type"], rel["head_description"]))
                else:
                    nodes_set.add((rel["head"], rel["head_type"]))
                if "tail_description" in rel.keys():
                    nodes_set.add((rel["tail"], rel["tail_type"], rel["tail_description"]))
                else:
                    nodes_set.add((rel["tail"], rel["tail_type"]))
                source_node = Node(id=rel["head"], type=rel["head_type"])
                target_node = Node(id=rel["tail"], type=rel["tail_type"])
                relationships.append(
                    Relationship(
                        source=source_node, target=target_node, type=rel["relation"]
                    )
                )
            nodes = []
            for el in list(nodes_set):
                if len(el) == 3:
                    node = Node(id=el[0], type=el[1], properties={"description": el[2]})
                else:
                    node = Node(id=el[0], type=el[1])
                nodes.append(node)
            
            return GraphDocument(nodes=nodes, relationships=relationships,source=source)
        except:
            self.logger.error(f"不是一个合法的Json")
            return None
    
    async def _create_nodes_and_relationships(self,documents:List[str]):
        data = self._preprocess(documents)
        documents = [Document(page_content=f"{row['title']} {row['text']}") for i, row in data.iterrows()]
        
        # 如果是openai模型，直接调用convert_to_graph_documents
        if isinstance(self.chat_model,ChatOpenAI):
            llm_transformer = LLMGraphTransformer(
                llm=self.chat_model,
                node_properties=["description"],
                relationship_properties=["description"],
                prompt=create_prompt(self.chat_model.name),
            )
            graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
        else:
            chat_prompt = create_prompt(self.chat_model.name)
            processed_documents = []
            for document in documents:
                prompt = chat_prompt.format_messages(input=document.page_content)
                processed_documents.append(self.chat_model.invoke(prompt))
            graph_documents = [self._parse_hf_ollama(document.content,source) for (document,source) in zip(processed_documents,documents)]
            graph_documents = [graph_document for graph_document in graph_documents if graph_document]
        
        return graph_documents
        
    async def create_index(self,documents:List[str]):
        self.logger.info("Create_nodes_and_relationships")
        graph_documents = await self._create_nodes_and_relationships(documents)

        for graph_document in graph_documents:
            for node in graph_document.nodes:
                node.type = node.type + f"{self.uuid}"

        for relationship in graph_document.relationships:
            relationship.type = relationship.type + f"{self.uuid}"
            relationship.source.type += f"{self.uuid}"
            relationship.target.type += f"{self.uuid}"

        # 将结点和关系存入图数据库
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        # 查询所有标签是__Entity__的结点，并修改成__Entity__+用户id
        self.cypherQuery.set_entity(self.uuid)
        # 查询所有标签是Document的结点，并修改成Document+用户id
        self.cypherQuery.set_document(self.uuid)
        
        self.graph.refresh_schema()
        Neo4jVector.from_existing_graph(
            self.embedding,
            node_label=f'__Entity__{self.uuid}',
            text_node_properties=['id', 'description'],
            index_name=f"{self.uuid}" if (self.uuid != None and self.uuid != "") else "vector",
            embedding_node_property='embedding',
            graph=self.graph,
        )
        try:
            self.cypherQuery.drop_entites()
        except:
            pass
        
        # 1.create the k-nearest neighbor graph
        G, _ = self.gds.graph.project(
            "entities",  # Graph name   # FIXME: 注册gds.graph时也要加上uuid,不然可能导致多进程误删除
            f"__Entity__{self.uuid}",  # Node projection
            "*",  # Relationship projection
            nodeProperties=["embedding"]  # Configuration parameters
        )
        # 2.algorithm: k-nearest neighbors
        self.gds.knn.mutate(
            G,
            nodeProperties=['embedding'],
            mutateRelationshipType='SIMILAR',
            mutateProperty='score',
            similarityCutoff=self.GDS_SIMILARITY_THRESHOLD,
        )
        # 3.store graph with weak connected components
        self.gds.wcc.write(
            G,
            writeProperty="wcc",
            relationshipTypes=["SIMILAR"]
        )
        # 4. KEY:社区检测与聚类分析
        
        potential_duplicate_candidates = self.cypherQuery.detect(self.uuid,self.WORD_EDIT_DISTANCE)
        extraction_llm = self.chat_model.with_structured_output(Disambiguate)
        extraction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SystemPrompts.IDENTIFY_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    UserPrompts.IDENTIFY_USER_PROMPT,   # noqa: E501,
                ),
            ]
        )
        extraction_chain = extraction_prompt | extraction_llm
        merged_entities = []
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(entity_resolution, el['combinedResult'],extraction_chain) for el in potential_duplicate_candidates]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
                try:
                    to_merge = future.result()
                    if to_merge:
                        merged_entities.extend(to_merge)
                except Exception as e:
                    self.logger.error("模型没法进行这条任务的实体解析")
        self.logger.info(countNodesMerged(self.uuid,merged_entities,self.graph))
        
        G.drop()
        
        self.cypherQuery.drop_communities()
        
        # 1.project into memory
        G, _ = self.gds.graph.project(
            f"communities",  # Graph name   # FIXME: 注册gds.graph时也要加上uuid,不然可能导致多进程误删除
            f"__Entity__{self.uuid}",  # Node projection
            {
                "_ALL_": {
                    "type": "*",
                    "orientation": "UNDIRECTED",
                    "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                }
            },
        )

        # 2. LeiDen聚类
        self.gds.leiden.write(
            G,
            writeProperty=f"communities",
            includeIntermediateCommunities=True,
            relationshipWeightProperty="weight",
        )
        
        # 添加约束
        self.cypherQuery.add_constraints_for_community(self.uuid)
        
        # 构造层次聚类
        merged_nodes = self.cypherQuery.constructing_hierarchical_clustering(self.uuid)
        self.logger.info(f"{merged_nodes[0]['count(*)']} nodes merged")
        
        # 设置社区rank
        self.cypherQuery.set_community_rank(self.uuid)
        
        # 设置结点与边的额外信息---用于后续的查询
        # 我们需要给所有实体结点设置度数；给边设置`source_degree`, `target_degree`, `rank`属性
        # 此外需要给每个实体设置其包含的text_unit_ids
        # 还需要给relationship设置source和target的属性，表示其链接到的结点的内容
        # 增加：需要给每个结点设置communities属性，是一个列表id，表示结点所在的社区
        # 1. node degree
        self.cypherQuery.set_node_degree(self.uuid)
        # 2. relationship degree
        self.cypherQuery.set_relationship_degree(self.uuid)
        # 3. text_unit_ids
        self.cypherQuery.set_text_unit_ids(self.uuid)
        # 4. relationship设置source和target的属性
        self.cypherQuery.set_relationship_source_and_target(self.uuid)
        # 5. 设置communities属性
        self.cypherQuery.set_communities(self.uuid)
        
        # 准备工作结束，开始summarization
        
        community_info = self.cypherQuery.get_community_info(self.uuid)

        
        community_chain = community_prompt | self.chat_model | StrOutputParser()    # TODO：增加报错处理
        summaries = []
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(process_communities, community, community_chain) for community in community_info}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
                summary = future.result()
                summaries.append(summary)
        
        
        title_chain = title_prompt | self.chat_model | StrOutputParser()
        titles = []
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(process_summaries, summary, title_chain) for summary in summaries}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Title"):
                title = future.result()
                titles.append(title)
        
        assert len(summaries) == len(titles)
        info = [{**summary, 'title': title} for summary, title in zip(summaries, titles)]
        
        # Store info
        self.cypherQuery.store_info(info,uuid=self.uuid)
        
        G.drop()