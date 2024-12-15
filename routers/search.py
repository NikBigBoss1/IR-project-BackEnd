import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from external_services.mongo_service import fetch_data
from external_services.pyterrier_service import initialize_pyterrier, search_documents, index_documents
from models.search_models import SearchRequest

router = APIRouter()

# Initialize PyTerrier when the server starts
initialize_pyterrier()
data = index_documents()

@router.post("/search")
def search(request: SearchRequest):
    results = search_documents(request.query, request.filters)
    if results.empty:
        raise HTTPException(status_code=404, detail="No results found")

    # Save results to search_results.csv
    file_path = "search_results.csv"
    results.to_csv(file_path, index=False)
    #return results.to_dict(orient="records")

    # Extract of top n results
    top_n_results = results.head(10)
    top_n_docnos = top_n_results["docno"].astype(str).tolist()
    top_n_events = data[data["docno"].isin(top_n_docnos)]

    events_file_path = "first_n_events.csv"
    top_n_events.to_csv(events_file_path, index=False)
    print(f"Results saved to {file_path}")


@router.post("/clustered-search")
def clustered_search(request: SearchRequest, cluster_by: str):
    """
    Perform a search and cluster the results based on a specified field.
    Args:
        request (SearchRequest): The search query and filters.
        cluster_by (str): The field to cluster by (e.g., 'Location', 'Venue', 'Genre').
    Returns:
        dict: Clustered search results with topics sorted by size.
    """
    # Perform the search
    results = search_documents(request.query, request.filters)
    if results.empty:
        raise HTTPException(status_code=404, detail="No results found")

    # Merge the results with the full dataset to access metadata fields
    results_with_metadata = pd.merge(results, data, on="docno", how="left")

    if cluster_by not in results_with_metadata.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cluster_by field: {cluster_by}. Available fields: {list(data.columns)}"
        )

    # Group by the chosen field
    grouped = results_with_metadata.groupby(cluster_by)

    # Create clusters
    clusters = [
        {
            "topic": cluster_name,
            "size": len(group),
            "results": group.to_dict(orient="records"),
        }
        for cluster_name, group in grouped
    ]

    # Sort clusters by size in descending order
    clustered_results = sorted(clusters, key=lambda x: x["size"], reverse=True)

    # Save to CSV
    save_clusters_to_csv(clustered_results)



def save_clusters_to_csv(clustered_results, output_file="clustered_results.csv"):
    # Flatten the clustered results into rows
    rows = []
    for cluster in clustered_results:
        topic = cluster["topic"]
        size = cluster["size"]
        for result in cluster["results"]:
            result_with_cluster = result.copy()
            result_with_cluster["topic"] = topic
            result_with_cluster["cluster_size"] = size
            rows.append(result_with_cluster)

    # Convert to a DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Clustered results saved to {output_file}")









@router.post("/cluster-collection")
def cluster_entire_collection(cluster_by: str):
    """
    Cluster the entire collection of items based on a specified field.
    Args:
        cluster_by (str): The field to cluster by (e.g., 'Location', 'Venue', 'Genre').
    Returns:
        dict: Clustered results with topics sorted by size.
    """

    if cluster_by not in data.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cluster_by field: {cluster_by}. Available fields: {list(data.columns)}"
        )

    # Group by the chosen field
    grouped = data.groupby(cluster_by)

    # Create clusters
    clusters = [
        {
               "topic": cluster_name,
               "size": len(group),
               "results": group.to_dict(orient="records"),
        }
        for cluster_name, group in grouped
      ]

    # Sort clusters by size in descending order
    sorted_clusters = sorted(clusters, key=lambda x: x["size"], reverse=True)

    # Save clusters to a CSV file
    output_file = "clustered_collection.csv"
    save_clusters_to_csv(sorted_clusters, output_file)



# @router.post("/cluster-collection")
# def cluster_entire_collection_endpoint(cluster_by: str, use_fuzzy: bool = False, use_tfidf: bool = False, n_clusters: int = 12, threshold: int = 80):
#     """
#     API endpoint to cluster the entire collection based on a specified field.
#     Args:
#         cluster_by (str): The field to cluster by (e.g., 'Location').
#         use_fuzzy (bool): Use fuzzy matching for clustering.
#         use_tfidf (bool): Use TF-IDF + KMeans for clustering.
#         n_clusters (int): Number of clusters (TF-IDF).
#         threshold (int): Similarity threshold for fuzzy matching.
#     Returns:
#         dict: Clustered results.
#     """
#     try:
#         # Fetch the entire dataset
#         data = fetch_data()
#
#         # Perform clustering
#         clustered_data = cluster_entire_collection(
#             data,
#             cluster_by=cluster_by,
#             use_fuzzy=use_fuzzy,
#             use_tfidf=use_tfidf,
#             n_clusters=n_clusters,
#             threshold=threshold
#         )
#
#         # Save to CSV
#         output_file = "clustered_collection.csv"
#         clustered_data.to_csv(output_file, index=False)
#         # return {
#         #     "message": f"Clustered collection saved to {output_file}",
#         #     "clusters": clustered_data.groupby("Cluster").size().to_dict()
#         # }
#
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
#
