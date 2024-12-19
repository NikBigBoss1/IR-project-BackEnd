import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pyterrier as pt

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

        # Vectorize the --------- LocationShort Cluster ------------ using TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(data["LocationShort"])

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=15, random_state=42)
        data["LocationShort Cluster"] = kmeans.fit_predict(tfidf_matrix)
        data["LocationShort Cluster"] = data["LocationShort Cluster"].astype(str)


        # Vectorize the --------- Price Cluster ------------ using TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(data["Price"])

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=6, random_state=42)
        data["Price Cluster"] = kmeans.fit_predict(tfidf_matrix)
        data["Price Cluster"] = data["Price Cluster"].astype(str)

        # Index documents with PyTerrier
        indexer = pt.IterDictIndexer(
            INDEX_DIR,
            meta=["docno", "cluster", "Event Name", "Date", "Venue", "Location", "LocationShort", "Price", "Description", "text", "Image Link", "Link", "LocationShort Cluster", "Price Cluster"]
        )
        documents = data[["docno", "text", "cluster", "Event Name", "Date", "Venue", "Location", "LocationShort", "Price", "Description", "Image Link", "Link", "LocationShort Cluster", "Price Cluster"]].to_dict(orient="records")
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
        eventName = [meta_index.getItem("Event Name", doc_id) for doc_id in doc_ids]
        date = [meta_index.getItem("Date", doc_id) for doc_id in doc_ids]
        venue = [meta_index.getItem("Venue", doc_id) for doc_id in doc_ids]
        location = [meta_index.getItem("Location", doc_id) for doc_id in doc_ids]
        price = [meta_index.getItem("Price", doc_id) for doc_id in doc_ids]
        description = [meta_index.getItem("Description", doc_id) for doc_id in doc_ids]
        imageLink = [meta_index.getItem("Image Link", doc_id) for doc_id in doc_ids]
        link = [meta_index.getItem("Link", doc_id) for doc_id in doc_ids]
        return pd.DataFrame({"docno": docnos, "cluster": clusters, "Event Name": eventName, "Date": date, "Venue": venue, "Location": location, "Price": price, "Description": description, "Image Link": imageLink, "Link": link})


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



def retrieve_data_by_cluster(cluster_id: int, cluster_type: str, index_dir: str) -> pd.DataFrame:
    """
    Retrieve all documents from a specific cluster.

    Args:
        cluster_id (int): The ID of the cluster to retrieve data for.
        cluster_type (str): The type of cluster column to filter by.
        index_dir (str): The path to the index directory.

    Returns:
        pd.DataFrame: DataFrame containing all documents from the specified cluster.
    """
    # Load the index and fetch metadata
    index = pt.IndexFactory.of(index_dir)
    meta_index = index.getMetaIndex()
    doc_ids = range(index.getCollectionStatistics().getNumberOfDocuments())

    # Retrieve metadata fields
    docnos = [meta_index.getItem("docno", doc_id) for doc_id in doc_ids]
    clusters = [meta_index.getItem(cluster_type, doc_id) for doc_id in doc_ids]
    eventName = [meta_index.getItem("Event Name", doc_id) for doc_id in doc_ids]
    date = [meta_index.getItem("Date", doc_id) for doc_id in doc_ids]
    venue = [meta_index.getItem("Venue", doc_id) for doc_id in doc_ids]
    location = [meta_index.getItem("Location", doc_id) for doc_id in doc_ids]
    price = [meta_index.getItem("Price", doc_id) for doc_id in doc_ids]
    description = [meta_index.getItem("Description", doc_id) for doc_id in doc_ids]
    imageLink = [meta_index.getItem("Image Link", doc_id) for doc_id in doc_ids]
    link = [meta_index.getItem("Link", doc_id) for doc_id in doc_ids]

    # Construct a DataFrame
    data = pd.DataFrame({"docno": docnos, cluster_type: clusters, "Event Name": eventName, "Date": date, "Venue": venue, "Location": location, "Price": price, "Description": description, "Image Link": imageLink, "Link": link})

    # Ensure the cluster_type column exists
    if cluster_type not in data.columns:
        raise ValueError(f"Cluster information ('{cluster_type}' column) is missing in the index.")

    # Filter by the specified cluster ID
    cluster_data = data[data[cluster_type] == str(cluster_id)]  # Ensure cluster_id matches as a string

    return cluster_data



def get_cluster_keywords(data: pd.DataFrame, text_col: str = "text", num_clusters: int = 5, num_keywords: int = 5) -> dict:
    """
    Create clusters and extract keywords for each cluster, applying Zipf's law and excluding the most frequent words.

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

    # Determine the most frequent words across all clusters
    global_word_counts = tfidf_matrix.sum(axis=0).A1
    most_frequent_words = set(feature_names[i] for i in global_word_counts.argsort()[-10:])  # Top 10 frequent words

    # Extract keywords for each cluster
    cluster_keywords = {}
    for cluster_id in range(num_clusters):
        cluster_data = tfidf_matrix[data["cluster"] == cluster_id]
        word_counts = cluster_data.sum(axis=0).A1

        # Apply Zipf's law: normalize by rank
        word_indices = word_counts.argsort()[::-1]  # Sort indices by descending frequency
        ranked_words = [(feature_names[i], word_counts[i] / (rank + 1)) for rank, i in enumerate(word_indices)]
        ranked_words = sorted(ranked_words, key=lambda x: x[1], reverse=True)

        # Exclude most frequent words
        filtered_words = [(word, score) for word, score in ranked_words if word not in most_frequent_words]

        # Select top keywords
        keywords = [word for word, score in filtered_words[:num_keywords]]
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

