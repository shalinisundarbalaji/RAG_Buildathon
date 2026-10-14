
from langchain_community.document_loaders import WebBaseLoader

# Agriculture-related website URLs
AGRICULTURE_URLS = [
    "https://www.fao.org/plant-production-protection/en/",
]

documents = []

for url in AGRICULTURE_URLS:
    try:
        loader = WebBaseLoader(web_paths=(url,))
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source_type"] = "URL"
            doc.metadata["source_url"] = url

        documents.extend(loaded_docs)
        print(f"Successfully loaded: {url}")

    except Exception as error:
        print(f"Failed to load {url}: {error}")

print("\nTotal webpages loaded:", len(documents))

if documents:
    print("\nSample webpage content:")
    print(documents[0].page_content[:1000])
