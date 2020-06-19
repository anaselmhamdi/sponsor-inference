from fastapi import FastAPI
from inference import handler
from pydantic import BaseModel

app = FastAPI()

class Inference(BaseModel):
    sentence: str

@app.post("/inference/")
def get_inference(s : Inference):
    return handler(s.sentence)
