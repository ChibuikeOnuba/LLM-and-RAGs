import os
from vectorizer import TFIDFVectorizer, SimpleTokenizer
from vectorstore import VectorStore


if __name__ == "__main__":

    folder = "C:/Users/onuba/Documents/LLM and RAGs/docs"
    documents =  []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                data = f.read()
                documents.append(data)

    doc_names = ["Google_Intro", "Mircrosoft_Intro", "Nvidia_Intro", "SpaceX_Intro", "Tesla_Intro"]

    #create vector store
    vector_store = VectorStore(chunk_size=50, overlap=10, max_features=200)

    vector_store.add_documents(documents, documents)

    #search
    # 
    print("\n" + "="*80)
    query = "what year was Google founded?"
    print(f"Question: {query}")
    print("="*80)


    answer = vector_store.search(query, top_k=3)

    for i, answer in enumerate(answer, 1):
        print(f"\nResult {i} (Score: {answer['score']:.4f})")

        print(f"Chunk: {answer['chunk'][:200]}...")
