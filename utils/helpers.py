# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
#
# def fuzzy_cluster_locations(data, field="Location", threshold=80):
#     """
#     Cluster similar location strings using fuzzy matching.
#     Args:
#         data (pd.DataFrame): The dataset containing location information.
#         field (str): The field to cluster (default: 'Location').
#         threshold (int): Similarity threshold (0-100) for grouping.
#     Returns:
#         pd.DataFrame: Original data with a 'Cluster' column indicating groups.
#     """
#     locations = data[field].tolist()
#     clusters = {}
#     cluster_id = 0
#
#     for location in locations:
#         matched = False
#         for cluster, members in clusters.items():
#             if fuzz.ratio(location, cluster) > threshold:
#                 clusters[cluster].append(location)
#                 matched = True
#                 break
#         if not matched:
#             clusters[location] = [location]
#             cluster_id += 1
#
#     # Create a mapping of location to cluster name
#     cluster_mapping = {
#         location: cluster for cluster, members in clusters.items() for location in members
#     }
#
#     # Assign clusters to the data
#     data["Cluster"] = data[field].map(cluster_mapping)
#     return data