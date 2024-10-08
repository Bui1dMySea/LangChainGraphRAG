from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

from ..custom_types.prompts import PromptBuilder


def _format_docs(documents: list[Document]) -> str:
    context_data = [d.page_content for d in documents]
    context_data_str: str = "\n".join(context_data)
    return context_data_str


class LocalSearch:
    def __init__(
        self,
        chat_model: BaseLLM,
        prompt_builder: PromptBuilder,
        retriever: BaseRetriever,
    ):
        self._chat_model = chat_model
        self._prompt_builder = prompt_builder
        self._retriever = retriever

    def __call__(self,query) -> Runnable:
        prompt, output_parser = self._prompt_builder.build()

        base_chain = prompt | self._chat_model | output_parser

        search_chain: Runnable = {
            "context_data": self._retriever | _format_docs,
            "local_query": RunnablePassthrough(),
        } | base_chain

        return search_chain.invoke(query)