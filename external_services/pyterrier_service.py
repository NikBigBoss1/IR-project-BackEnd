import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pyterrier as pt

from external_services.mongo_service import fetch_data
from settings import INDEX_DIR

def initialize_pyterrier():
    """
    Initialize PyTerrier.
    """
    if not pt.started():
        pt.init()




def preprocess_and_cluster(data: pd.DataFrame, num_clusters: int = 5, text_col: str = "text") -> pd.DataFrame:
    """
    Preprocess text data and apply clustering.

    Args:
        data (pd.DataFrame): The dataset to cluster.
        num_clusters (int): Number of clusters to generate.
        text_col (str): The column containing text data for clustering.

    Returns:
        pd.DataFrame: The dataset with an additional 'cluster' column.
    """
    if text_col not in data.columns:
        raise ValueError(f"Column '{text_col}' not found in the dataset.")

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(data[text_col])

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data["cluster"] = kmeans.fit_predict(tfidf_matrix)

    return data


def index_documents(data: pd.DataFrame):
    """
    Index the documents into PyTerrier.

    Args:
        data (pd.DataFrame): The dataset to index.

    Returns:
        None
    """
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Convert 'cluster' to string, as PyTerrier expects metadata as strings
        data["cluster"] = data["cluster"].astype(str)

        # Index documents with PyTerrier
        indexer = pt.IterDictIndexer(
            INDEX_DIR,
            meta=["docno", "cluster", "Event Name", "Date", "Venue", "Location", "LocationShort", "Price", "Description"]
        )
        documents = data[["docno", "text", "cluster", "Event Name", "Date", "Venue", "Location", "LocationShort", "Price", "Description", "Image Link", "Link"]].to_dict(orient="records")
        indexer.index(documents)
        print("Indexing completed successfully!")
    else:
        print("Indexing already completed.")


def search_documents(query: str = "", filters=None, cluster_id=None) -> pd.DataFrame:
    """
    Search for documents using BM25 and apply optional filters or cluster constraints.

    Args:
        query (str): The search query.
        filters (list): Optional filters to apply to the query.
        cluster_id (int): Cluster ID to restrict results to.

    Returns:
        pd.DataFrame: Search results.
    """
    if not pt.started():
        pt.init()

    def fetch_metadata(index_dir: str) -> pd.DataFrame:
        """Fetch metadata (docno and cluster) from the index."""
        index = pt.IndexFactory.of(index_dir)
        meta_index = index.getMetaIndex()
        doc_ids = range(index.getCollectionStatistics().getNumberOfDocuments())

        docnos = [meta_index.getItem("docno", doc_id) for doc_id in doc_ids]
        clusters = [meta_index.getItem("cluster", doc_id) for doc_id in doc_ids]
        return pd.DataFrame({"docno": docnos, "cluster": clusters})


    if not query.strip():  # If the query is empty
        print("No query provided, fetching entire dataset")

        data = fetch_metadata(INDEX_DIR)

        if filters:
            query = " " + " AND ".join(filters)
            print(f"Composed query: {query}")

            bm25 = pt.BatchRetrieve(INDEX_DIR, wmodel="BM25")
            results = bm25.search(query)

            # Filter results by cluster_id if provided
            if cluster_id is not None:
                print(f"Filtering search results by cluster_id: {cluster_id}")

                metadata = fetch_metadata(INDEX_DIR)
                results = pd.merge(results, metadata, on="docno", how="left")
                results = results[results["cluster"] == str(cluster_id)]

            return results

        # If cluster_id is provided, filter by cluster_id
        if cluster_id is not None:
            print(f"Filtering by cluster_id: {cluster_id}")
            data = data[data["cluster"] == str(cluster_id)]

        return data

    else:
        bm25 = pt.BatchRetrieve(INDEX_DIR, wmodel="BM25")

        # Add filters to the query
        if filters:
            query += " " + " AND ".join(filters)

        results = bm25.search(query)

        # Filter results by cluster_id if provided
        if cluster_id is not None:
            print(f"Filtering search results by cluster_id: {cluster_id}")

            metadata = fetch_metadata(INDEX_DIR)
            results = pd.merge(results, metadata, on="docno", how="left")
            results = results[results["cluster"] == str(cluster_id)]

        return results



def retrieve_data_by_cluster(cluster_id: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve all documents from a specific cluster.

    Args:
        cluster_id (int): The ID of the cluster to retrieve data for.
        data (pd.DataFrame): The dataset containing the 'cluster' column.

    Returns:
        pd.DataFrame: DataFrame containing all documents from the specified cluster.
    """
    # Ensure the 'cluster' column exists
    if "cluster" not in data.columns:
        raise ValueError("Cluster information ('cluster' column) is missing. Ensure clustering is applied first.")

    # Filter by the specified cluster ID
    cluster_data = data[data["cluster"] == cluster_id]

    return cluster_data



def get_cluster_keywords(data: pd.DataFrame, text_col: str = "text", num_clusters: int = 5, num_keywords: int = 5) -> dict:
    """
    Create clusters and extract keywords for each cluster.

    Args:
        data (pd.DataFrame): Dataset containing the text data.
        text_col (str): Column containing the text data to be clustered.
        num_clusters (int): Number of clusters to generate.
        num_keywords (int): Number of keywords to extract per cluster.

    Returns:
        dict: A dictionary mapping cluster IDs to lists of keywords.
    """
    # Preprocess and cluster data
    data = preprocess_and_cluster(data, num_clusters=num_clusters, text_col=text_col)

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(data[text_col])
    feature_names = vectorizer.get_feature_names_out()

    # Extract keywords for each cluster
    cluster_keywords = {}
    for cluster_id in range(num_clusters):
        cluster_data = tfidf_matrix[data["cluster"] == cluster_id]
        word_counts = cluster_data.sum(axis=0).A1
        keywords = [feature_names[i] for i in word_counts.argsort()[-num_keywords:][::-1]]
        cluster_keywords[cluster_id] = keywords

    return cluster_keywords


def save_clusters_to_csv(data: pd.DataFrame, output_file: str = "clustered_results.csv"):
    """
    Save clustered data to a CSV file.

    Args:
        data (pd.DataFrame): Data with cluster labels.
        output_file (str): Path to save the clustered results CSV.
    """
    data.to_csv(output_file, index=False)
    print(f"Clustered results saved to {output_file}")

