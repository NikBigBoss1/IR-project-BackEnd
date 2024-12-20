import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import pyterrier as pt
from external_services.mongo_service import fetch_data
from external_services.pyterrier_service import (
    initialize_pyterrier,
    preprocess_and_cluster,
    index_documents,
    search_documents,
    get_cluster_keywords,
    save_clusters_to_csv, retrieve_data_by_cluster,
)
from models.search_models import ClusterRequest, SearchRequest
from settings import INDEX_DIR

# Initialize PyTerrier when the server starts
initialize_pyterrier()

data = fetch_data()
clustered_data = preprocess_and_cluster(data, num_clusters=15, text_col="text")
index_documents(clustered_data)


# Initialize Router
router = APIRouter()


@router.post("/cluster-data")
def cluster_data(request: ClusterRequest):
    """
    Preprocess data and create clusters.
    """
    try:
        data = fetch_data()
        clustered_data = preprocess_and_cluster(data, num_clusters=request.num_clusters, text_col=request.text_col)

        sorted_data = clustered_data.sort_values(by="cluster").reset_index(drop=True)
        save_clusters_to_csv(sorted_data, output_file="clustered_data.csv")
        return {"message": "Clustering completed successfully.", "clusters_saved_to": "clustered_data.csv"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search(request: SearchRequest):
    """
    Perform a search using PyTerrier with optional filters and cluster constraints.
    """
    try:
        results = search_documents(request.query, filters=request.filters, cluster_id=request.cluster_id)

        index = pt.IndexFactory.of(INDEX_DIR)
        meta_index = index.getMetaIndex()
        doc_ids = range(index.getCollectionStatistics().getNumberOfDocuments())

        # Retrieve all metadata fields
        metadata_fields = [
            "docno", "Event Name", "Date", "Venue", "Location",
            "Price", "Description", "Image Link", "Link", "cluster"
        ]

        metadata = {field: [meta_index.getItem(field, doc_id) for doc_id in doc_ids] for field in metadata_fields}
        metadata_df = pd.DataFrame(metadata)

        full_results = pd.merge(results, metadata_df, on="docno", how="left")

        if results.empty:
            raise HTTPException(status_code=404, detail="No results found.")

        return {"message": "Search completed successfully.", "results": full_results.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cluster/{cluster_id}")
def get_data_by_cluster(cluster_id: int, cluster_type: str):
    """
    Retrieve all data from a specific cluster.

    Args:
        cluster_id (int): The ID of the cluster to retrieve data for.
        cluster_type (str): The type of cluster column to filter by.

    Returns:
        dict: A dictionary containing the documents from the specified cluster.
    """
    try:
        # Retrieve data for the specified cluster and type
        data_from_cluster = retrieve_data_by_cluster(cluster_id, cluster_type, INDEX_DIR)

        if data_from_cluster.empty:
            raise HTTPException(status_code=404, detail=f"No data found for cluster ID {cluster_id}.")

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



@router.get("/export-indexed-data")
def export_indexed_data():
    """
    Export all indexed data to a CSV file.
    Test router to see what data we have, as indexed data

    Returns:
        dict: Success message with the output file path.
    """
    try:
        index = pt.IndexFactory.of(INDEX_DIR)
        meta_index = index.getMetaIndex()
        doc_ids = range(index.getCollectionStatistics().getNumberOfDocuments())

        # List of metadata fields to fetch
        metadata_fields = [
            "Event Name", "Date", "Venue", "Location",
            "Price", "Description", "Image Link", "Link", "docno", "cluster", "text", "LocationShort", "LocationShort Cluster", "Price Cluster"
        ]

        metadata = {}
        for field in metadata_fields:
            try:
                metadata[field] = [meta_index.getItem(field, doc_id) for doc_id in doc_ids]
            except Exception as e:
                print(f"Warning: Unable to retrieve field '{field}': {e}")
                metadata[field] = [None] * len(doc_ids)

        indexed_data = pd.DataFrame(metadata)

        # Save the DataFrame to a CSV file
        output_file = "indexedData.csv"
        indexed_data.to_csv(output_file, index=False)
        print(f"Indexed data exported to {output_file}")

        return {"message": "Indexed data exported successfully.", "output_file": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while exporting indexed data: {str(e)}")


@router.get("/locations")
def get_locations():
    """
    Retrieve all unique locations from the LocationShort field in the indexed data.

    Returns:
        dict: A list of unique locations.
    """
    try:
        index = pt.IndexFactory.of(INDEX_DIR)
        meta_index = index.getMetaIndex()
        doc_ids = range(index.getCollectionStatistics().getNumberOfDocuments())

        location_short = [meta_index.getItem("LocationShort", doc_id) for doc_id in doc_ids]

        unique_locations = sorted(set(filter(None, location_short)))

        return {"locations": unique_locations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching locations: {str(e)}")



@router.get("/venue")
def get_venues():
    """
    Retrieve all unique locations from the LocationShort field in the indexed data.

    Returns:
        dict: A list of unique locations.
    """
    try:
        index = pt.IndexFactory.of(INDEX_DIR)
        meta_index = index.getMetaIndex()
        doc_ids = range(index.getCollectionStatistics().getNumberOfDocuments())

        location_short = [meta_index.getItem("Venue", doc_id) for doc_id in doc_ids]

        unique_locations = sorted(set(filter(None, location_short)))

        return {"venue": unique_locations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching locations: {str(e)}")


@router.get("/event/{docno}")
def get_event_by_docno(docno: str):
    """
    Retrieve an event from the indexed data based on the given docno.

    Args:
        docno (str): The unique document number.

    Returns:
        dict: The event data corresponding to the given docno.
    """
    try:
        index = pt.IndexFactory.of(INDEX_DIR)
        meta_index = index.getMetaIndex()
        doc_ids = range(index.getCollectionStatistics().getNumberOfDocuments())

        docnos = [meta_index.getItem("docno", doc_id) for doc_id in doc_ids]

        if docno not in docnos:
            raise HTTPException(status_code=404, detail=f"Document with docno '{docno}' not found.")

        # Retrieve the corresponding document metadata
        index_doc_id = docnos.index(docno)
        event = {
            "docno": docno,
            "text": meta_index.getItem("text", index_doc_id),
            "cluster": meta_index.getItem("cluster", index_doc_id),
            "Event Name": meta_index.getItem("Event Name", index_doc_id),
            "Date": meta_index.getItem("Date", index_doc_id),
            "Venue": meta_index.getItem("Venue", index_doc_id),
            "Location": meta_index.getItem("Location", index_doc_id),
            "Price": meta_index.getItem("Price", index_doc_id),
            "Description": meta_index.getItem("Description", index_doc_id),
            "Image Link": meta_index.getItem("Image Link", index_doc_id),
            "Link": meta_index.getItem("Link", index_doc_id),
        }

        return {"message": "Event retrieved successfully.", "event": event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving event: {str(e)}")