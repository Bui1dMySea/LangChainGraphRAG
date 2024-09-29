from pathlib import Path

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

from ...custom_types.prompts import PromptBuilder

from ._system_prompt import REDUCE_SYSTEM_PROMPT


class KeyPointsAggregatorPromptBuilder(PromptBuilder):
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        system_prompt_path: Path | None = None,
    ):
        self._system_prompt: str | None
        if system_prompt is None and system_prompt_path is None:
            self._system_prompt = REDUCE_SYSTEM_PROMPT
        else:
            self._system_prompt = system_prompt

        self._system_prompt_path = system_prompt_path

    def build(self) -> tuple[BasePromptTemplate, BaseOutputParser]:
        if self._system_prompt_path:
            if type(self._system_prompt_path) is str:
                self._system_prompt_path = Path(self._system_prompt_path)
            prompt = self._system_prompt_path.read_text(encoding='utf-8')
        else:
            assert self._system_prompt is not None
            prompt = self._system_prompt

        system_template = SystemMessagePromptTemplate.from_template(
            prompt,
            partial_variables=dict(response_type="Multiple Paragraphs"),
        )

        template = ChatPromptTemplate([system_template, ("user", "{global_query}")])
        return template, StrOutputParser()