from fastapi import FastAPI
from transformers import pipeline

from api.schemas import SentimentRequest, SentimentResponse


MODEL_PATH = "./model/sentiment_model"


app = FastAPI(
    title="IMDb Sentiment Analyzer",
    description="DistilBERT-based sentiment classification API",
    version="1.0.0",
)


classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
)


@app.get("/")
def root():
    return {
        "message": "IMDb Sentiment Analyzer API",
        "status": "running",
    }


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):

    result = classifier(request.text)[0]

    label = result["label"]
    confidence = float(result["score"])

    sentiment = {
        "LABEL_0": "negative",
        "LABEL_1": "positive",
    }.get(label, label)

    return SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
    )