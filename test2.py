from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
llm = HuggingFacePipeline.from_model_id(
    model_id="NousResearch/Meta-Llama-3.1-8B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=True,
        repetition_penalty=1.03,
        return_full_text=False
    ),
    device=0,
)
llm.pipeline.tokenizer.pad_token_id = llm.pipeline.tokenizer.eos_token_id 

chat_model = ChatHuggingFace(llm=llm)
breakpoint()
messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="Tell me about 3 technological companies!Please only answer the name without any other information."
    ),
]

ai_msg = chat_model.invoke(messages)