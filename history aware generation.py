from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

#take user input
print("Enter query")
query = str(input())

#Create a chatOpenAI model
model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(user_question):
    print(f"\n---------- You asked {user_question}-----------------")

    if chat_history:
        #ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and seachable. Just return the rewritten question")
            ] + chat_history + [
                HumanMessage(content=f"New Question: {user_question}")
            ]
        
        result = model.invoke(messages)
        search_question = result.content.strip() if isinstance(result.content, str) else str(result.content).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
        

    #search vector store
    retriever = db.as_retriever(search_kwargs={"k":3})
    relevant_docs = retriever.invoke(search_question)

    print("\n----------------------------Context---------------------------")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content[100]}\n")

    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {'chr(10)'.join([f"-{doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the inforamtion from these documents. If you can't find the answer in the documents, say "I dont have the answer based on the provided documents"""


    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)

    #Display the full result and content only
    print("\n -------------------Generated Respone-----------------------------")

    print(f"Content: {result.content}")