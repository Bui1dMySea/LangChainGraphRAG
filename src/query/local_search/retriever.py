from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from .context_builders import ContextBuilder
from .context_selectors import ContextSelector


class LocalSearchRetriever(BaseRetriever):
    context_selector: ContextSelector
    context_builder: ContextBuilder
    graph: VectorStore
    # artifacts: IndexerArtifacts

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        context_selection_result = self.context_selector.run(
            query=query,
            # artifacts=self.artifacts,
            graph=self.graph,
        )

        return self.context_builder(context_selection_result)