python search.py \
--neo4j_uri bolt://localhost:7687 \
--neo4j_username neo4j \
--neo4j_password  langchaingraphrag \
--model_provider openai \
--embedding_model_name_or_path BAAI/bge-m3 \
--chat_model_name gpt-4o-mini \
--base_url https://api.gpt.ge/v1/ \
--completion_mode completion \
--query_mode global