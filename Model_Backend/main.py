from pydantic import BaseModel
from fastapi import FastAPI , HTTPException, Query, status,Response
from typing import Optional
import uvicorn
from Model import Model
from DataBase import DBConnection

url = "https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf"

app = FastAPI()
model = Model()
db = DBConnection()

class File(BaseModel):
    document_id : str

class Question(BaseModel):
    document_id: str
    question: str

data = {
    "name" :"umair hassan",
    "roll Number":"BSEF20M537"
}

@app.get('/')
def test():
    return {"message" :"running perfectly good again"}

@app.post('/upload',status_code=status.HTTP_201_CREATED)
def uploading_document(file : File):
    print("coming in")
    file = db.get_file(file.document_id)
    try:
        down_time , db_time = model.store_file(str(file["_id"]) , file["fileData"])
        if down_time == -1:
            return {
                "message" : "collection already exists"
            }
        return {
            "message" : "downloaded and stored",
            "down_time" : down_time,
            "db_time" : db_time
        }  
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400 , detail="Error in uploading file boss")


@app.post("/ask_query")
def ask_query(question : Question):
    response = model.ask_model(question.document_id , question.question)
    return {
        "model response" : response[0],
        "db time" : response[1],
        "model time" : response[2]
    }