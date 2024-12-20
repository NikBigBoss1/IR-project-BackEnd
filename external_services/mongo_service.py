from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "events_db"
COLLECTION_NAME = "eventsTest"

def fetch_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    data = pd.DataFrame(list(collection.find({}, {"_id": 0})))
    data = data.fillna("")

    # Add a unique document number
    data['docno'] = [str(i) for i in range(len(data))]

    # Add cluster
    data['cluster'] = [None]*len(data)

    # Create a combined 'text' field for indexing
    data['text'] = (
            data['Event Name'] + " " +
            data['Description'] + " " +
            data['Price'] + " " +
            data['Venue'] + " " +
            data['Location']
    )

    # -----------------Add short text field LocationShort
    cities_df = pd.read_csv("gb.csv")
    valid_cities = set(cities_df["city"].dropna().astype(str))
    valid_admin_names = set(cities_df["admin_name"].dropna().astype(str))

    valid_locations = valid_cities.union(valid_admin_names)

    def extract_valid_cities(location):
        if pd.isna(location):
            return ""
        location_str = str(location)
        matched_cities = [city for city in valid_locations if city in location_str]
        return ", ".join(matched_cities) if matched_cities else ""

    data["LocationShort"] = data["Location"].apply(extract_valid_cities)


    return data