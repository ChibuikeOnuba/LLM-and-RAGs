from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

#Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#search through the relevant documents
print("Enter query")
query = str(input())

retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(search_type="similarity_score_threshold",
#         search_kwargs={
#             "k": 5,
#             "score_threshold": 0.3}
#             )


relevant_docs = retriever.invoke(query)

print(f"User Input: {query}")
print("\n----------------------------Context---------------------------")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{'chr(10)'.join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the inforamtion from these documents. If you can't find the answer in the documents, say "I dont have the answer based on the provided documents"""

#Create a chatOpenAI model
model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

#Display the full result and content only
print("\n -------------------Generated Respone-----------------------------")

print(f"Content: {result.content}")

print("\n -------------------Token Usage-----------------------------")

if hasattr(result, 'usage_metadata') and result.usage_metadata is not None:
    prompt_tokens = result.usage_metadata.get('prompt_tokens', 0)
    completion_tokens = result.usage_metadata.get('completion_tokens', 0)
    total_tokens = result.usage_metadata.get('total_tokens', 0)
    
    # Calculate cost (GPT-4 pricing example)
    input_cost = prompt_tokens * 0.00003  # $0.03 per 1K tokens
    output_cost = completion_tokens * 0.00006  # $0.06 per 1K tokens
    total_cost = input_cost + output_cost
    
    print(f"Cost for this query: ${total_cost:.4f}")
    print(f"Tokens used: {total_tokens}")