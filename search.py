import os

from src.query import LocalSearcher, GlobalSearcher
from langchain_community.graphs import Neo4jGraph
from langchain_openai.chat_models import ChatOpenAI

from src.utils.logger import create_rotating_logger
import argparse
import logging
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
import getpass

@dataclass
class LOG_LEVELS:
    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL

def parse_args():
    arg_parser = argparse.ArgumentParser(description="search for LangChainGraphRAG")
    
    arg_parser.add_argument("--neo4j_uri", type=str,default=None, help="Neo4j URI")
    arg_parser.add_argument("--neo4j_username", type=str,default=None, help="Neo4j user")
    arg_parser.add_argument("--neo4j_password", type=str,default=None, help="Neo4j password")
    # FIXME:目前仅支持openai
    arg_parser.add_argument("--model_provider", type=str,choices=['openai','ollama'], help="Model provider")
    arg_parser.add_argument("--chat_model_name", type=str, help="Chat model name")
    # arg_parser.add_argument("--api_key", type=str, help="API key")
    arg_parser.add_argument("--base_url", type=str, help="Base URL")
    
    # FIXME:可以做更好的区分；embedding可以使用bge-m3等模型也可以用api模型
    arg_parser.add_argument("--embedding_model_name_or_path", type=str, help="Embedding Model name")
    
    arg_parser.add_argument("--uuid", type=str, default="", help="UUID for the search")
    arg_parser.add_argument("--top_k", type=int, default=15,help="top_k for the search")
    arg_parser.add_argument("--level", type=int, default=1, help="level for the search")
    arg_parser.add_argument("--max_tokens", type=int, default=8000, help="Max tokens")
    
    arg_parser.add_argument("--log_file", type=str, default="search.log", help="Log file")
    arg_parser.add_argument("--log_level", type=str, default="info", choices=['debug','info','warning','error','critical'],help="Log level")
    # arg_parser.add_argument("--max_workers", type=int, default=4, help="Max workers")
    # arg_parser.add_argument("--device", type=str, default="cpu",choices=['cuda','cpu'],help="Device")

    arg_parser.add_argument("--completion_mode",type=str,choices=['chat','completion'],default='chat',help="完成模式")
    arg_parser.add_argument("--query_mode",type=str,choices=['local','global'],default='local',help="查询模式")
    
    return arg_parser.parse_args()


def query():
    args = parse_args()
    
    log_level = getattr(LOG_LEVELS, args.log_level)
    logger = create_rotating_logger("search", args.log_file, level=log_level)
    
    # logging args
    logger.info(f"args: {args}")
    
    # FIXME:目前得确保model_provider为openai
    if args.model_provider != "openai":
        logger.error("目前只支持openai")
        return
    
    # 初始化环境变量
    # 优先从os.environ中获取，如果没有则从args中获取
    if os.environ.get("NEO4J_URI") is None:
        os.environ["NEO4J_URI"] = args.neo4j_uri
    if os.environ.get("NEO4J_USERNAME") is None:
        os.environ["NEO4J_USERNAME"] = args.neo4j_username
    if os.environ.get("NEO4J_PASSWORD") is None:
        os.environ["NEO4J_PASSWORD"] = args.neo4j_password

    logger.info("Connecting to Neo4j")
    
    try:
        graph = Neo4jGraph()
    except:
        logger.error(
                     "Failed to connect to Neo4j"
                     f"URI: {args.uri}, Username: {args.username}, Password: Your password"
        )
        return
    
    # 初始化chat model
    if os.environ.get("OPENAI_API_KEY") is None:
        try:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        except:
            logger.error("Failed to get OpenAI API key")
            return
    
    logger.info("Initializing chat model")
    chat_model = ChatOpenAI(model=args.chat_model_name, base_url=args.base_url, api_key=os.environ["OPENAI_API_KEY"])
    # 初始化embedding
    logger.info("Initializing embedding")
    # model_kwargs = {'device': args.device}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceEmbeddings(model_name=args.embedding_model_name_or_path,encode_kwargs=encode_kwargs,show_progress=True)
    logger.info("查询模式：{}".format(args.query_mode))
    if args.query_mode == 'local':
        searcher = LocalSearcher(
            graph=graph,
            chat_model=chat_model,
            embedding=embedding,
            uuid=args.uuid,
            top_k=args.top_k,
            level=args.level
        )
    elif args.query_mode == 'global':
        searcher = GlobalSearcher(
            graph=graph,
            chat_model=chat_model,
            uuid=args.uuid,
            level=args.level,
            max_tokens=args.max_tokens
        )
    
    completion_mode = args.completion_mode

    logger.info("Starting query now!")
    # 查询逻辑
    if completion_mode == 'chat':
        try:
            while True:
                query = input("请输入您的查询 (输入 'exit' 或 'quit' 退出):\n")
                if query.lower() in ['exit', 'quit']:
                    logger.info("退出程序")
                    break
                else:
                    # 在这里处理用户的查询
                    result = searcher.invoke(query)
                    logger.info(f"查询结果:\n{result}")
        except KeyboardInterrupt:
            print("\n程序被中断，退出。")
                
    elif completion_mode == 'completion':
        query = input("请输入您的查询:\n")
        result = searcher.invoke(query)
        logger.info(f"查询结果:\n{result}")

    logger.info("查询结束")
    
if __name__ == "__main__":
    query()
    
    
    
