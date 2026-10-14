
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the agriculture PDF
pdf_path = "agriculture_pdfs/Agriculture_Guide.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("PDF pages loaded:", len(documents))

# 2. Split PDF content into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=600
)

chunks = text_splitter.split_documents(documents)

print("Total chunks created:", len(chunks))

if not chunks:
    raise ValueError("No text chunks found in the PDF.")

# 3. Create local embeddings
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store embeddings in FAISS
print("Creating FAISS vector database...")

vector_db = FAISS.from_documents(chunks, embeddings)

# 5. Save the vector database
vector_db.save_local("agriculture_faiss_index")

print("FAISS vector database created successfully!")
print("Saved to: agriculture_faiss_index")
