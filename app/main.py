from fastapi import FastAPI
from inference import handler
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentenceInference(BaseModel):
    sentence: str

class VideoInference(BaseModel):
    youtube_url: str

@app.post("/inference/sentence")
def get_sentence_inference(s : SentenceInference):
    return handler(s.sentence, None)

@app.post("/inference/video_url")
def get_video_inference(i : VideoInference):
    return handler(None, i.youtube_url)
