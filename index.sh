python build_index.py \
--file_path /home/kas/kas_workspace/liuweijie/LangChainGraphRAG/txt \
--neo4j_uri bolt://localhost:7687 \
--neo4j_username neo4j \
--neo4j_password lwj@wpsai \
--model_provider openai \
--embedding_model_name_or_path /home/kas/kas_workspace/liuweijie/bge-base-zh-v1.5 \
--chat_model_name deepseek-chat \
--base_url https://api.deepseek.com \
--max_workers 16 \
--device cuda