
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the same embedding model used during indexing
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load the saved FAISS database
vector_db = FAISS.load_local(
    "agriculture_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Ask a question
query = input("Ask an agriculture question: ")

# 4. Retrieve the most relevant chunks
results = vector_db.similarity_search(query, k=3)

# 5. Display the results
print("\n--- RETRIEVED AGRICULTURE INFORMATION ---")

for i, document in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Source:", document.metadata.get("source", "Unknown"))
    print("Content:")
    print(document.page_content)
    print("-" * 50)
