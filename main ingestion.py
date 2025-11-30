import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#document loader function
def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")


    if not os.path.exists(docs_path):
        raise FileExistsError(f"The directory {docs_path} does not exist. Please create and add your files")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls= TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileExistsError(f"No files found. Please ensure you have .txt files in your folder {docs_path}")
    else:
        print(f"{len(documents)} documents loaded")

    # for i, doc in enumerate(documents[:2]):
    #     print(f"\nDocument {i+1}:")
    #     print(f" Source: {doc.metadata['source']}")
    #     print(f" Content Length: {len(doc.page_content)} characters")
    #     print(f" Content Preview: {doc.page_content[:200]}...")
    #     print(f" Metadata: {doc.metadata}")

    return documents


#document Chunker function
def split_documents(documents, chunk_size=800, chunk_overlap=50):
    print("Splitting the documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


    chunks = text_splitter.split_documents(documents)

    # if chunks:

    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n --- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content: ")
    #         print(chunk.page_content)
    #         print("-" *50)

    #     if len(chunks) > 5:
    #         print(f"\n ... and {len(chunks) - 5} more chunks")

    return chunks

#Vector Store function
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model= "text-embedding-3-small")

    #create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"--- Vector Store Created and saved to {persist_directory} ---")

    return vectorstore

def main():

    #1. Load Documents
    documents = load_documents(docs_path="docs")

    #2. Chunk
    chunks = split_documents(documents)
 
    #3. Load to vector db
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()