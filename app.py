
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="Agriculture RAG Assistant",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Agriculture RAG Assistant")
st.write("Search information from your indexed PDF and agriculture websites.")

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "agriculture_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

try:
    vector_db = load_vector_db()

    question = st.text_input(
        "Ask an agriculture question",
        placeholder="Example: What does FAO do to protect crops?"
    )

    if st.button("Search"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the vector database..."):
                results = vector_db.similarity_search_with_score(
                    question, k=3
                )

            st.subheader("Retrieved information")

            if not results:
                st.info("No matching documents were retrieved.")
            else:
                for i, (doc, score) in enumerate(results, start=1):
                    with st.expander(f"Result {i}", expanded=True):
                        st.write(doc.page_content)
                        st.caption(
                            f"Source: {doc.metadata.get('source_url', doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown')))}"
                        )
                        st.caption(f"FAISS distance: {score:.4f}")

except Exception as error:
    st.error(f"Could not load the vector database: {error}")
