from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex, set_global_service_context
from llama_index.llms import LlamaCPP

llm = LlamaCPP()

# llm = HuggingFaceLLM(
#     context_window=4096,
#     max_new_tokens=2048,
#     generate_kwargs={"temperature": 0.0, "do_sample": False},
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name=selected_model,
#     model_name=selected_model,
#     device_map="auto",
#     # change these settings below depending on your GPU
#     model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
# )

EXCLUDE = ['*.png', '*jpg', '*.jpeg']
documents = SimpleDirectoryReader("./data", recursive=True, exclude=EXCLUDE).load_data(show_progress=True)

print(f'Loaded {len(documents)} documents.')

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en"
)
set_global_service_context(service_context)

index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist()
