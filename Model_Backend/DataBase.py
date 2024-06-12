import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId

class DBConnection:
    def __init__(self) -> None:
        load_dotenv()
        self.connection_string = os.environ.get("MONGO_URI")
        print(f"connection string  = {self.connection_string}")
        client = MongoClient(self.connection_string)
        db = client["test"]
        self.collection = db["files"]
    
    def get_file(self, file_id: str):
        file = self.collection.find_one({"_id": ObjectId(file_id)})
        return file

export = DBConnection()