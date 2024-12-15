import pyterrier as pt
import os
from settings import INDEX_DIR

from external_services.mongo_service import fetch_data

def initialize_pyterrier():
    if not pt.started():
        pt.init()

def index_documents():

    data = fetch_data()

    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR, exist_ok=True)

        indexer = pt.IterDictIndexer(
            INDEX_DIR,
            meta=["docno"],  # Only 'docno' is required as metadata
        )
        documents = data[['docno', 'text', 'Event Name', 'Date', 'Venue', 'Location', 'Price', 'Description', 'Image Link', 'Link']].to_dict(orient='records')
        indexer.index(documents)
        print("Indexing completed successfully!")
    else:
        print("Indexing already was completed")

    return data

def search_documents(query: str, filters=None):
    bm25 = pt.terrier.Retriever(INDEX_DIR, wmodel="BM25")
    if filters:
        query += " " + " AND ".join(filters)
    return bm25.search(query)