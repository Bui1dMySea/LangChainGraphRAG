from .utils.token_counter import TiktokenCounter
from .local_search.context_builders import ContextBuilder
from .local_search.context_selectors import ContextSelector
from .local_search.retriever import LocalSearchRetriever
from .local_search.search import LocalSearch
from .local_search.prompt_builder import LocalSearchPromptBuilder
from .global_search import key_points_generator
from .global_search import key_points_aggregator
from .global_search.search import GlobalSearch
from .global_search.community_weight_calculator import CommunityWeightCalculator

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from typing import Literal
class LocalSearcher(object):
    def __init__(
                    self,
                    graph:Neo4jGraph,
                    chat_model:BaseChatModel,
                    embedding:Embeddings, 
                    uuid:str= None,
                    top_k:int=15,
                    level:int=1,
                    # model_provider:Literal['openai','ollama','hf']='openai',
                    # model_name:str,
                    # api_key:str,
                    # base_url:str,
                    *args,
                    **kwargs
                 ):
        
        token_counter = TiktokenCounter() 
        vector_store = Neo4jVector.from_existing_graph(
                embedding=embedding,
                index_name=f"{uuid}" if (uuid != None and uuid != "") else "vector",
                node_label=f'__Entity__{uuid}',
                text_node_properties=['id','description'],
                embedding_node_property='embedding',
                graph=graph
        )
        
        context_builder = ContextBuilder.build_default(token_counter)
        context_selector = ContextSelector.build_default(vector_store, top_k, level, uuid)

        # chat_model = ChatOpenAI(model=model_name,base_url=base_url, api_key=api_key)
        retriever = LocalSearchRetriever(
            context_selector=context_selector,
            context_builder=context_builder,
            graph=vector_store
        )
        self.local_search = LocalSearch(chat_model=chat_model, prompt_builder=LocalSearchPromptBuilder(), retriever=retriever)

    def invoke(self,query:str):
        return self.local_search(query)

class GlobalSearcher(object):
    def __init__(
                self,
                graph:Neo4jGraph,
                chat_model:BaseChatModel,
                uuid:str=None,
                level:str=1,
                max_tokens:int=8000,
                # model_provider:Literal['openai','ollama'],
                # model_name:str,
                # api_key:str,
                # base_url:str,
                *args,
                **kwargs
            ):
        cwc = CommunityWeightCalculator()
        token_counter = TiktokenCounter()
        kpg_prompt_builder = key_points_generator.KeyPointsGeneratorPromptBuilder()
        kpg_context_builder = key_points_generator.CommunityReportContextBuilder(level, cwc, uuid, graph,token_counter,max_tokens)

        kpa_prompt_builder = key_points_aggregator.KeyPointsAggregatorPromptBuilder()
        kpa_context_builder = key_points_aggregator.KeyPointsContextBuilder(token_counter)

        kp_aggregator = key_points_aggregator.KeyPointsAggregator(chat_model, kpa_prompt_builder, kpa_context_builder)
        kp_generator = key_points_generator.KeyPointsGenerator(chat_model, kpg_prompt_builder, kpg_context_builder)
    
        self.global_search = GlobalSearch(kp_generator, kp_aggregator)
        

    def invoke(self,query:str):
        return self.global_search.invoke(query)
        
        