from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

chat_history = []

print("Bot: Hello! Ask me anything about your documents. (Type 'quit' to exit)\n")

while True:
    # Get user input
    query = input("You: ").strip()
    
    # Check exit condition FIRST
    if query.lower() in ['quit', 'exit', 'q']:
        print("Bot: Goodbye!")
        break
    
    # Skip empty inputs
    if not query:
        continue
    
    print(f"\n---------- You asked: \"{query}\" -----------------")

    # Determine the question to search for
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=f"New Question: {query}")
        ]
        
        result = model.invoke(messages)
        user_question = result.content.strip() if isinstance(result.content, str) else str(result.content).strip()
        print(f"Rewritten as: {user_question}")
    else:
        user_question = query

    print(f"Searching for: {user_question}")

    # Search vector store
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(user_question)

    print("\n----------------------------Context---------------------------")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content[:200]}...\n")

    # Combine documents and create prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have the answer based on the provided documents"."""

    # Generate response
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)

    # Display the response
    print("\n-------------------Generated Response-----------------------------")
    print(f"Bot: {result.content}\n")

    # UPDATE CHAT HISTORY - this is the key fix!
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result.content))