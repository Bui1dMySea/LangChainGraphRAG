python build_index.py \
--file_path ./txt \
--neo4j_uri bolt://localhost:7687 \
--neo4j_username neo4j \
--neo4j_password  langchaingraphrag \
--model_provider ollama \
--chat_model_name llama3.1 \
--embedding_model_name_or_path BAAI/bge-m3 \
--max_workers 16 \
--device cuda
