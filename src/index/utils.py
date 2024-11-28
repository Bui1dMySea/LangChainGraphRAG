from typing import List, Optional
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document

from retry import retry
import numpy as np
import tiktoken
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import asyncio

from .prompts import SystemPrompts, UserPrompts



def num_tokens_from_string(string: str, model: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def visualizeEntityTokenDistibution(graph: Neo4jGraph,user_id):
    entity_dist = graph.query(
       f"""
    MATCH (d:Document{user_id})
    RETURN d.text AS text,
        count {{(d)-[:MENTIONS]->()}} AS entity_count
    """
    )
    entity_dist_df = pd.DataFrame.from_records(entity_dist)
    entity_dist_df["token_count"] = [
        num_tokens_from_string(str(el)) for el in entity_dist_df["text"]
    ]
    # Scatter plot with regression line
    sns.lmplot(
        x="token_count", y="entity_count", data=entity_dist_df, line_kws={"color": "red"}
    )
    plt.title("Entity Count vs Token Count Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Entity Count")
    plt.show()
    plt.savefig('entity_token_distribution.png')


def visualizeCommunityEntityDistribution(graph: Neo4jGraph, user_id):
    # 查询每个层次的社区包含的实体的数量
    community_size = graph.query(f"""
        MATCH (c:__Community__{user_id})<-[:IN_COMMUNITY*]-(e:__Entity__{user_id}) 
        WITH c, count(distinct e) AS entities   
        RETURN split(c.id, '-')[0] AS level, entities
        """
    )
    
    community_size_df = pd.DataFrame.from_records(community_size)
    
    # 计算百分位数
    percentiles_data = []
    for level in community_size_df['level'].unique():
        subset = community_size_df[community_size_df['level'] == level]['entities']
        num_communities = len(subset)
        percentiles = np.percentile(subset, [25, 50, 75, 90, 99])
        percentiles_data.append(
            [
                level,
                num_communities,
                percentiles[0],
                percentiles[1],
                percentiles[2],
                percentiles[3],
                percentiles[4],
                max(subset),
            ]
        )
    
    percentiles_df = pd.DataFrame(
        percentiles_data,
        columns=[
            "Level",
            "Num Communities",
            "25th Percentile",
            "50th Percentile",
            "75th Percentile",
            "90th Percentile",
            "99th Percentile",
            "Max Communities",
        ],
    )
    
    # 创建图形和子图
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # 可视化最大社群数量
    sns.barplot(data=percentiles_df, x='Level', y='Max Communities', ax=axs[0], color='skyblue', label='Max Communities', alpha=0.7)
    sns.barplot(data=percentiles_df, x='Level', y='50th Percentile', ax=axs[0], color='orange', label='50th Percentile', alpha=0.5)
    axs[0].set_title('Community Entity Distribution by Level', fontsize=16)
    axs[0].set_ylabel('Number of Communities', fontsize=14)
    axs[0].legend()
    axs[0].grid(axis='y')

    # 可视化社群个数
    sns.barplot(data=percentiles_df, x='Level', y='Num Communities', ax=axs[1], color='lightgreen')
    axs[1].set_title('Number of Communities by Level', fontsize=16)
    axs[1].set_xlabel('Community Level', fontsize=14)
    axs[1].set_ylabel('Number of Communities', fontsize=14)
    axs[1].grid(axis='y')

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存集成的图像为 PNG 文件
    plt.savefig(f'community_distribution_combined_{user_id}.png', dpi=300)
    plt.show()


def countNodesMerged(user_id,merged_entities,graph: Neo4jGraph):
    count = graph.query("""
        UNWIND $data AS candidates
        CALL {{
            WITH candidates
            MATCH (e:{label}) WHERE e.id IN candidates
            RETURN collect(e) AS nodes
        }}
        CALL apoc.refactor.mergeNodes(nodes, {{properties: {{`.*`: 'discard'}}}})
        YIELD node
        RETURN count(*)
        """.format(label=f"__Entity__{user_id}"), params={"data": merged_entities}
    )
    print(f"{count} nodes merged")

def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"
    rels_str = "Relationships are:\n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str

def create_prompt(model_name):
    system_prompt = SystemPrompts.GRAPHSYSTEMPROMPT.format(model_name=model_name)
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessagePromptTemplate.from_template(UserPrompts.GRAPH_USER_PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    return chat_prompt


async def aprocess_summaries(summaries, title_chain):
    tasks = [asyncio.create_task(title_chain.ainvoke({"summary": summary})) for summary in summaries]
    results = await asyncio.gather(*tasks)
    return results

def process_summaries(summary, title_chain):
    result = title_chain.invoke({"summary": summary})
    return result

async def aprocess_communities(community_info, community_chain):
    string_info_list = [prepare_string(community) for community in community_info]
    tasks = [asyncio.create_task(community_chain.ainvoke({'community_info': string_info})) for string_info in string_info_list]
    results = await asyncio.gather(*tasks)
    info_summary = []
    for community, result in zip(community_info, results):
        summary = result.output
        info_summary.append(
            {"community": community['communityId'], "summary": summary})
    return info_summary

def process_communities(community, community_chain):
    stringify_info = prepare_string(community)
    summary = community_chain.invoke({'community_info': stringify_info})
    return {"community": community['communityId'], "summary": summary}


def process_text(text: str, model) -> List[GraphDocument]:
    doc = Document(page_content=text)
    return model.convert_to_graph_documents([doc])


async def aprocess_text(texts: List[str], model) -> List[GraphDocument]:
    docs = [Document(page_content=text) for text in texts]
    return await model.aconvert_to_graph_documents(docs)

@retry(tries=3, delay=2)
async def aentity_resolution(entities: List[str], extraction_chain) -> Optional[List[str]]:
    results = await extraction_chain.ainvoke({"entities": entities})
    return [el.entities for el in results.merge_entities]


@retry(tries=3, delay=2)
def entity_resolution(entities: List[str], extraction_chain) -> Optional[List[str]]:
    return [el.entities for el in extraction_chain.invoke({"entities": entities}).merge_entities]

