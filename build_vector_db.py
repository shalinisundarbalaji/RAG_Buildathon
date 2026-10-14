
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load PDF documents
pdf_path = "agriculture_pdfs/Agriculture_Guide.pdf"

pdf_loader = PyPDFLoader(pdf_path)
pdf_documents = pdf_loader.load()

print("PDF pages loaded:", len(pdf_documents))

# 2. Load webpage content
url = "https://www.fao.org/plant-production-protection/en/"

web_loader = WebBaseLoader(web_paths=(url,))
web_documents = web_loader.load()

for doc in web_documents:
    doc.metadata["source_type"] = "URL"
    doc.metadata["source_url"] = url

print("Web documents loaded:", len(web_documents))

# 3. Combine both sources
all_documents = pdf_documents + web_documents

print("Total documents:", len(all_documents))

# 4. Split content into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_documents)

print("Total chunks created:", len(chunks))

if not chunks:
    raise ValueError("No chunks were created. Check your source documents.")

# 5. Generate embeddings
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 6. Create the vector database
print("Creating FAISS vector database...")

vector_db = FAISS.from_documents(chunks, embeddings)

# 7. Save the database
vector_db.save_local("agriculture_faiss_index")

print("\nSUCCESS: Vector database created!")
print("Location: agriculture_faiss_index")
