import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from external_services.mongo_service import fetch_data
from external_services.pyterrier_service import (
    initialize_pyterrier,
    preprocess_and_cluster,
    index_documents,
    search_documents,
    get_cluster_keywords,
    save_clusters_to_csv, retrieve_data_by_cluster,
)

# Initialize PyTerrier when the server starts
initialize_pyterrier()

data = fetch_data()
clustered_data = preprocess_and_cluster(data, num_clusters=9, text_col="Location")
index_documents(clustered_data)


# Initialize Router
router = APIRouter()



# Pydantic Models
class SearchRequest(BaseModel):
    query: str
    filters: Optional[List[str]] = None
    cluster_id: Optional[int] = None


class ClusterRequest(BaseModel):
    text_col: str
    num_clusters: int
    num_keywords: Optional[int] = 5


# @router.get("/fetch-data")
# def fetch_and_return_data():
#     """
#     Fetch data from MongoDB.
#     """
#     try:
#         data = fetch_data()
#         return {"message": "Data fetched successfully.", "data_count": len(data)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster-data")
def cluster_data(request: ClusterRequest):
    """
    Preprocess data and create clusters.
    """
    try:
        data = fetch_data()
        clustered_data = preprocess_and_cluster(data, num_clusters=request.num_clusters, text_col=request.text_col)

        # Step 2: Sort by cluster ID
        sorted_data = clustered_data.sort_values(by="cluster").reset_index(drop=True)
        save_clusters_to_csv(sorted_data, output_file="clustered_data.csv")
        return {"message": "Clustering completed successfully.", "clusters_saved_to": "clustered_data.csv"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# @router.post("/index-data")
# def index_data():
#     """
#     Index data into PyTerrier after clustering.
#     """
#     try:
#         data = fetch_data()
#         clustered_data = preprocess_and_cluster(data, num_clusters=5, text_col="text")
#         index_documents(clustered_data)
#         return {"message": "Indexing completed successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search(request: SearchRequest):
    """
    Perform a search using PyTerrier with optional filters and cluster constraints.
    """
    try:
        results = search_documents(request.query, filters=request.filters, cluster_id=request.cluster_id)
        if results.empty:
            raise HTTPException(status_code=404, detail="No results found.")
        save_clusters_to_csv(results, output_file="data_on_specific_cluster.csv")

        # Extract of top n results
        top_n_results = results.head(10)
        top_n_docnos = top_n_results["docno"].astype(str).tolist()
        top_n_events = data[data["docno"].isin(top_n_docnos)]
        save_clusters_to_csv(top_n_events, output_file="data_on_specific_cluster_exactly.csv")

        return {"message": "Search completed successfully.", "results": results.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cluster/{cluster_id}")
def get_data_by_cluster(cluster_id: int):
    """
    Retrieve all data from a specific cluster.

    Args:
        cluster_id (int): The ID of the cluster to retrieve data for.

    Returns:
        dict: A dictionary containing the documents from the specified cluster.
    """
    try:

        # Retrieve data for the specified cluster
        data_from_cluster = retrieve_data_by_cluster(cluster_id, clustered_data)

        if data_from_cluster.empty:
            raise HTTPException(status_code=404, detail=f"No data found for cluster ID {cluster_id}.")

        save_clusters_to_csv(data_from_cluster, output_file="specific_cluster_data.csv")
        return {
            "message": "Data retrieved successfully.",
            "cluster_id": cluster_id,
            "data": data_from_cluster.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/cluster-keywords")
def extract_cluster_keywords(request: ClusterRequest):
    """
    Extract keywords for clusters from the dataset.
    """
    try:
        keywords = get_cluster_keywords(data, text_col=request.text_col, num_clusters=request.num_clusters, num_keywords=request.num_keywords)
        return {"message": "Keywords extracted successfully.", "keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))