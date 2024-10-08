from typing import List,Optional,Any,Union,Literal
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
# from algo.chunk.utils import pretty_print
import re


def _split_text_with_regex(
    text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1]
                 for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                (splits + [_splits[-1]])
                if keep_separator == "end"
                else ([_splits[0]] + splits)
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class SentenceSlidingWindowChunkSplitter(TextSplitter):
    def __init__(
        self,
        sliding_chunk_size: int,
        separators: Optional[List[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = False,
        is_separator_regex: bool = False,
        sliding_distance: int = 2,
        **kwargs: Any,
    ) -> None:
        # 不需要chunk overlap
        super().__init__(keep_separator=keep_separator, chunk_overlap=0, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        self.sliding_distance = sliding_distance
        self.sliding_chunk_size = sliding_chunk_size
        # self.tokenzier = kwargs.get("tokenizer") if kwargs.get("tokenizer") else None
        assert (
            self.sliding_distance >= 0
        ), "Sliding distance must be greater than or equal to 0."
        if self._chunk_size > self.sliding_chunk_size:
            Warning(
                "Chunk size is bigger than sliding chunk_size, setting chunk size to sentence size."
            )
            self._chunk_size = self.sliding_chunk_size

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(
            separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
        """Text splitter that uses HuggingFace tokenizer to count length."""
        try:
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Tokenizer received was not an instance of PreTrainedTokenizerBase"
                )

            def _huggingface_tokenizer_length(text: str) -> int:
                return len(tokenizer.encode(text))

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
        return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    # Core function
    def split_text(self, text: str) -> List[str]:
        sentence_chunks = self._split_text(text, self._separators)
        final_chunks = []
        # 合并
        for i in range(len(sentence_chunks)):
            combined_split = sentence_chunks[i]
            j = 1

            while j <= self.sliding_distance:
                if i - j >= 0:
                    if (
                        self._length_function(
                            sentence_chunks[i - j] + combined_split)
                        > self.sliding_chunk_size
                    ):
                        break
                    combined_split = sentence_chunks[i - j] + combined_split
                if i + j < len(sentence_chunks):
                    if (
                        self._length_function(
                            combined_split + sentence_chunks[i + j])
                        > self.sliding_chunk_size
                    ):
                        break
                    combined_split += sentence_chunks[i + j]
                j += 1
            final_chunks.append(combined_split)
        return final_chunks