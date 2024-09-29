import asyncio
from typing import Sequence
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from tqdm.asyncio import tqdm

class t_LLMGraphTransformer(LLMGraphTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def aconvert_to_graph_documents(self, documents: Sequence[Document], config: Optional[RunnableConfig] = None) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = [
            asyncio.create_task(self.aprocess_response(document, config))
            for document in documents
        ]
        results = await tqdm.gather(*tasks)
        return results