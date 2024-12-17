from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "events_db"
COLLECTION_NAME = "eventsTest"

def fetch_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    # Retrieve data from the MongoDB collection
    data = pd.DataFrame(list(collection.find({}, {"_id": 0})))  # Exclude _id field
    data = data.fillna("")  # Replace null values with empty strings

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
    return data