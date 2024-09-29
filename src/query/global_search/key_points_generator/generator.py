from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import Runnable, RunnableParallel

from ...custom_types.prompts import PromptBuilder

from .context_builder import CommunityReportContextBuilder

import json_repair
import json

def _format_docs(documents: list[Document]) -> str:
    context_data = [d.page_content for d in documents]
    context_data_str: str = "\n".join(context_data)
    return context_data_str

class KeyPointsGenerator:
    def __init__(
        self,
        llm: BaseLLM,
        prompt_builder: PromptBuilder,
        context_builder: CommunityReportContextBuilder,
    ):
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._context_builder = context_builder

    # 生成关键点
    def __call__(self) -> Runnable:
        prompt, output_parser = self._prompt_builder.build()
        documents = self._context_builder()

        chains: list[Runnable] = []
        for d in documents:
            d_context_data = _format_docs([d])
            d_prompt = prompt.partial(context_data=d_context_data)
            
            # TODO:异常处理
            generator_chain: Runnable = d_prompt | self._llm | (lambda output:json.dumps(json_repair.loads(output.content))) | output_parser
            
            chains.append(generator_chain)  

        analysts = [f"Analayst-{i}" for i in range(1, len(chains) + 1)]
        # {"Analyst-1": "Chain-1", "Analyst-2": "Chain-2", "Analyst-3": "Chain-3"}
        return RunnableParallel(dict(zip(analysts, chains, strict=True)))