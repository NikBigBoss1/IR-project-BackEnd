from pymongo import MongoClient
import pandas as pd
import os
from decouple import config

MONGO_URI = config('MONGO_URI')
DB_NAME = "events_db"
COLLECTION_NAME = "eventsTest"

def fetch_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    data = pd.DataFrame(list(collection.find({}, {"_id": 0})))  # Exclude _id
    data = data.fillna("")  # Replace null values with empty strings

    # Add 'docno' and combined 'text' fields
    data['docno'] = [str(i) for i in range(len(data))]
    data['text'] = (
        data['Event Name'] + " " + data['Price'] + " " + data['Venue'] + " " + data['Location']
    )
    return data