import os
import pandas as pd
from typing import  List
from tqdm.asyncio import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer

# utils
from .utils import num_tokens_from_string,create_prompt,process_text,entity_resolution,process_communities,process_summaries
from graphdatascience import GraphDataScience
from .cypher_query import CypherQuery

# doc2ppt
from algo.chunk.slide_window_method import SentenceSlidingWindowChunkSplitter
from const import env
from algo.embedding.embed import LangChainEmbeddings

# prompt
from .prompts import SystemPrompts, UserPrompts
# hf
from transformers import AutoTokenizer
# pydantic models
from .pydantic_models import Disambiguate, DuplicateEntities, GetTitle

# 获取当前文件（kdb_operation.py）的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取core目录的绝对路径
core_directory = os.path.abspath(os.path.join(os.path.dirname(current_file_path), '../../..'))
# 构建到tokenizer目录的绝对路径
tokenizer_dir_path = os.path.join(core_directory, 'algo', 'embedding', 'tokenizer')

def llm_create_index(documents: List[str],graph:Neo4jGraph, user_id: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_path)
    splitter = SentenceSlidingWindowChunkSplitter.from_huggingface_tokenizer(tokenizer, sliding_chunk_size=500, sliding_distance=0)
    cypherQuery = CypherQuery(graph=graph)
    data = []
    for doc in documents:
        chunks = []
        parts = doc.split('\n')
        title = parts[0]
        texts = '\n'.join(parts[1:])
        chunks.extend(splitter.split_text(texts))
        for chunk in chunks:
            data.append({"title": title, "text": chunk})
    # df
    df_data = pd.DataFrame(data)
    texts = [f"{row['title']} {row['text']}" for index,row in df_data.iterrows()]
    openai = ChatOpenAI(model=env.MODEL_NAME,base_url=env.BASE_URL, api_key=env.API_KEY)
    llm_transformer = LLMGraphTransformer(
        llm=openai,
        node_properties=["description"],
        relationship_properties=["description"],
        prompt=create_prompt(env.MODEL_NAME),
    )
    # total_tokens = sum([num_tokens_from_string(text) for text in texts])
    
    MAX_WORKERS = env.MAX_WORKERS
    
    graph_documents = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submitting all tasks and creating a list of future objects
        futures = [executor.submit(process_text, f"{row['title']} {row['text']}", llm_transformer) for i, row in df_data.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            asyn_graph_document = future.result()
            graph_documents.extend(asyn_graph_document)
    
    for graph_document in graph_documents:
        for node in graph_document.nodes:
            node.type = node.type + f"__{user_id}"
            # node.properties["user_id"] = user_id
        for relationship in graph_document.relationships:
            relationship.type = relationship.type + f"_{user_id}"
            relationship.source.type += f"_{user_id}"
            relationship.target.type += f"_{user_id}"
    
    # 将结点和关系存入图数据库
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    
    # 查询所有标签是__Entity__的结点，并修改成__Entity__+用户id
    cypherQuery.set_entity(user_id)
    
    # 查询所有标签是Document的结点，并修改成Document+用户id
    cypherQuery.set_document(user_id)
    
    graph.refresh_schema()
    
    # FIXME：项目中每个用户需要关闭实例后需要删除索引
    # graph.query(f"DROP INDEX `{user_id}` IF EXISTS")
    # graph.query(f"MATCH (n) WHERE ANY(label IN labels(n) WHERE label ENDS WITH '__{user_id}')")

    url = env.BGE_EMBEDDING_URL
    
    embedding = LangChainEmbeddings(url)
    
    Neo4jVector.from_existing_graph(
        embedding,
        node_label=f'__Entity__{user_id}',
        text_node_properties=['id', 'description'],
        index_name=f"{user_id}",
        embedding_node_property='embedding',
        graph=graph,
    )
    
    # project graph
    # Graph Data Science (GDS) library
    gds = GraphDataScience(
        env.NEO4J_URI,
        auth=(env.NEO4J_USER, env.NEO4J_PASSWORD)
    )
    
    cypherQuery.drop_entites()
    
    # 1.create the k-nearest neighbor graph
    G, result = gds.graph.project(
        "entities",  # Graph name
        f"__Entity__{user_id}",  # Node projection
        "*",  # Relationship projection
        nodeProperties=["embedding"]  # Configuration parameters
    )
    # 2.algorithm: k-nearest neighbors
    gds.knn.mutate(
        G,
        nodeProperties=['embedding'],
        mutateRelationshipType='SIMILAR',
        mutateProperty='score',
        similarityCutoff=env.GDS_SIMILARITY_THRESHOLD,
    )
    # 3.store graph with weak connected components
    gds.wcc.write(
        G,
        writeProperty="wcc",
        relationshipTypes=["SIMILAR"]
    )
    # 4. KEY:社区检测与聚类分析
    word_edit_distance = 3
    potential_duplicate_candidates = cypherQuery.detect(user_id,word_edit_distance)
    extraction_llm = openai.with_structured_output(Disambiguate)
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
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submitting all tasks and creating a list of future objects
        futures = [executor.submit(entity_resolution, el['combinedResult'],extraction_chain) for el in potential_duplicate_candidates]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            try:
                to_merge = future.result()
                if to_merge:
                    merged_entities.extend(to_merge)
            except Exception as e:
                print("模型没法进行这条任务的实体解析")
    
    # countNodesMerged(user_id,merged_entities,graph)
    
    G.drop()
    
    cypherQuery.drop_communities()
    
    # 1.project into memory
    G, result = gds.graph.project(
        f"communities",  # Graph name
        f"__Entity__{user_id}",  # Node projection
        {
            "_ALL_": {
                "type": "*",
                "orientation": "UNDIRECTED",
                "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
            }
        },
    )
    
    # 查看图连通性
    # wcc = gds.wcc.stats(G)
    # print(f"Component count: {wcc['componentCount']}")
    # print(f"Component distribution: {wcc['componentDistribution']}")
    
    # 2. LeiDen聚类
    gds.leiden.write(
        G,
        writeProperty=f"communities",
        includeIntermediateCommunities=True,
        relationshipWeightProperty="weight",
    )
    
    # 添加约束
    cypherQuery.add_constraints_for_community(user_id)
    
    # 构造层次聚类
    merged_nodes = cypherQuery.constructing_hierarchical_clustering(user_id)
    print(f"{merged_nodes[0]['count(*)']} nodes merged")
    
    # 设置社区rank
    cypherQuery.set_community_rank(user_id)
    
    # 设置结点与边的额外信息---用于后续的查询
    # 我们需要给所有实体结点设置度数；给边设置`source_degree`, `target_degree`, `rank`属性
    # 此外需要给每个实体设置其包含的text_unit_ids
    # 还需要给relationship设置source和target的属性，表示其链接到的结点的内容
    # 增加：需要给每个结点设置communities属性，是一个列表id，表示结点所在的社区
    # 1. node degree
    cypherQuery.set_node_degree(user_id)
    # 2. relationship degree
    cypherQuery.set_relationship_degree(user_id)
    # 3. text_unit_ids
    cypherQuery.set_text_unit_ids(user_id)
    # 4. relationship设置source和target的属性
    cypherQuery.set_relationship_source_and_target(user_id)
    # 5. 设置communities属性
    cypherQuery.set_communities(user_id)
    
    # 准备工作结束，开始summarization
    
    community_info = cypherQuery.get_community_info(user_id)

    community_template = """Based on the provided nodes and relationships that belong to the same graph community,
    generate a natural language summary of the provided information:
    {community_info}

    Summary:"""  # noqa: E501

    community_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input triples, generate the information summary. No pre-amble.",
            ),
            ("human", community_template),
        ]
    )
    
    community_chain = community_prompt | openai | StrOutputParser()
    summaries = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_communities, community, community_chain) for community in community_info}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            summary = future.result()
            summaries.append(summary)
    
    
    title_template = """Given the following summary, provide a title that best represents the content:
    {summary}
    
    Title:"""
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a summary, generate a title that best represents the content. No pre-amble.",
            ),
            ("human", title_template),
        ]
    )
    
    title_chain = title_prompt | openai | StrOutputParser()
    titles = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_summaries, summary, title_chain) for summary in summaries}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Title"):
            title = future.result()
            titles.append(title)
    
    assert len(summaries) == len(titles)
    info = [{**summary, 'title': title} for summary, title in zip(summaries, titles)]
    
    # Store info
    cypherQuery.store_info(user_id,info)
    
    G.drop()