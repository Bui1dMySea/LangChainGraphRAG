import os
import pathlib
import asyncio
from src.index.api_index import ApiIndex
from src.splitter.slide_window_splitter import SentenceSlidingWindowChunkSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline,HuggingFaceEmbeddings,ChatHuggingFace
from langchain.schema.output_parser import BaseOutputParser
from langchain_ollama import ChatOllama
from src.utils.logger import create_rotating_logger
from graphdatascience import GraphDataScience
import logging
import argparse
import getpass
import re
from dataclasses import dataclass
import torch
  
class JSONDelimiterParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Try to find JSON content within ```json and ``` delimiters
        match = re.search(r'```json(.*?)```', text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            # If delimiters are missing, assume the entire text is JSON
            json_content = text.strip()
        # Return the JSON content wrapped in delimiters
        return '```json\n' + json_content + '\n```'


@dataclass
class LOG_LEVELS:
    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL

def parse_args():
    arg_parser = argparse.ArgumentParser(description="Build index for LangChainGraphRAG")
    arg_parser.add_argument("--file_path", type=str, help="File path")
    arg_parser.add_argument("--neo4j_uri", type=str,default=None, help="Neo4j URI")
    arg_parser.add_argument("--neo4j_username", type=str,default=None, help="Neo4j user")
    arg_parser.add_argument("--neo4j_password", type=str,default=None, help="Neo4j password")
    arg_parser.add_argument("--uuid", type=str, default="", help="UUID for the index")
    arg_parser.add_argument("--model_provider", type=str,choices=['openai','ollama','hf'], help="Model provider")
    arg_parser.add_argument("--embedding_model_name_or_path", type=str, help="embedding")
    # arg_parser.add_argument("--api_key", type=str, help="API key")
    arg_parser.add_argument("--base_url", type=str, help="Base URL")
    # hf
    arg_parser.add_argument("--repo_id", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct",help="Repo ID")
    arg_parser.add_argument("--chat_model_name", type=str, help="Model name")
    arg_parser.add_argument("--flash_attn",action="store_true",help="where use Flash attention or not ")
    arg_parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens")
    arg_parser.add_argument("--topK", type=int, default=50, help="Top K for hf model initialization")
    
    arg_parser.add_argument("--top_k", type=int, default=15, help="Top K for graphrag")
    arg_parser.add_argument("--level", type=int, default=1, help="Level")
    arg_parser.add_argument("--log_file", type=str, default="index.log", help="Log file")
    arg_parser.add_argument("--log_level", type=str, default="info", choices=['debug','info','warning','error','critical'],help="Log level")
    arg_parser.add_argument("--max_workers", type=int, default=4, help="Max workers")
    arg_parser.add_argument("--device", type=str, default="cpu",choices=['cuda','cpu'],help="Device")
    arg_parser.add_argument("--chunk_size",type=int,default=500,help="Chunk size")
    return arg_parser.parse_args()

async def build_index():
    args = parse_args()
    log_level = getattr(LOG_LEVELS, args.log_level)
    logger = create_rotating_logger("build_index", args.log_file, level=log_level)
    logger.info("Start building index")
    
    # 获取file_path下所有的txt文件
    file_path = pathlib.Path(args.file_path)
    if not file_path.exists():
        logger.error(f"File path {args.file_path} does not exist")
        return
    files = [str(file) for file in file_path.glob("*.txt")]
    if not files:
        logger.error(f"请检查文件路径正确并且检查{args.file_path}下是否有txt文件；目前只支持txt文件")
        return
    # 读取文件内容
    documents = []
    
    for file in files:
        with open(file, "r") as f:
            # 获取file的名字，不要文件格式
            file_name = file.split("/")[-1].split(".")[0]
            documents.append({"title":file_name,"text":f.read()})
    # 确保documents不为空
    if not documents:
        logger.error("找不到内容，请确保文件目录下的文件不为空")
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
    except Exception as e:
        logger.error(
                     f"Meet error {e} when connecting to Neo4j \n" 
                     "Failed to connect to Neo4j"
                     f"URI: {args.neo4j_uri}, Username: {args.neo4j_username}, Password: {args.neo4j_password}"
        )
        return
    
    
    # 初始化chat model
    if os.environ.get("OPENAI_API_KEY") is None and args.model_provider == "openai":
        try:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        except:
            logger.error("Failed to get OpenAI API key")
            return
    
    logger.info("Initializing chat model")
    if args.model_provider == "openai":
        chat_model = ChatOpenAI(model=args.chat_model_name, base_url=args.base_url, api_key=os.environ["OPENAI_API_KEY"])
    elif args.model_provider == "hf":
        llm = HuggingFacePipeline.from_model_id(
            model_id=args.repo_id,
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                # repetition_penalty=1.03,
                return_full_text=False,
                # top_k=args.topK,
            ),
            model_kwargs=dict(
                attn_implementation="flash_attention_2" if args.flash_attn else None,
                torch_dtype=torch.bfloat16,
            ),
            device=0,
            # device_map="auto" if args.device == "cuda" else None,
        )
        # parser = JSONDelimiterParser()
        llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id 
        chat_model = ChatHuggingFace(llm=llm)
    elif args.model_provider == "ollama":
        chat_model = ChatOllama(
            model=args.chat_model_name,
            num_predict=512,
            temperature=0.0,
        )

    # 初始化embedding
    logger.info("Initializing embedding")
    # model_kwargs = {'device': args.device}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceEmbeddings(model_name=args.embedding_model_name_or_path,encode_kwargs=encode_kwargs,show_progress=True)
    # 初始化splitter
    logger.info("Initializing splitter")
    splitter = SentenceSlidingWindowChunkSplitter(args.chunk_size)
    # 初始化gds
    logger.info("Initializing GDS")
    gds = GraphDataScience(os.environ["NEO4J_URI"],auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    # 初始化index
    logger.info("Initializing index")
    index = ApiIndex(
        graph=graph,
        chat_model=chat_model,
        embedding=embedding,
        gds=gds,
        uuid=args.uuid,
        # top_k=args.top_k,
        # level=args.level,
        # max_tokens=args.max_tokens,
        splitter=splitter,
        max_workers=args.max_workers,
        logger=logger,
        model_name=args.chat_model_name
    )
    # 构建索引
    logger.info("Building index now!")
    await index.create_index(documents=documents)
    logger.info("Index built successfully")

if __name__ == "__main__":
    asyncio.run(build_index())
