from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
from llama_index.llms import LlamaCPP

llm = LlamaCPP()

SYSTEM_PROMPT = """You were quantizied on a XML file named model.xml.
This XML is a Enterprise Architect export of a UML model.
You should answer any question based on the data from this model.
"""

# Set the Service Context (prevent tokenzier from accessing OpenAI)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en"
)
set_global_service_context(service_context)

# Load the quantized index from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# Prepare query engine
query_engine = index.as_query_engine(system_prompt=SYSTEM_PROMPT)

while True:
    question = input("Ask me anything: ")
    response = query_engine.query(question)
    print(f"<b>{response}</b>")
