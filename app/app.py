from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import time
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

class Ticket(BaseModel):

    subject: str
    body: str
    language: str = "en"
    priority: int = 2

class ClusterResponse(BaseModel):

    cluster_id: int
    cluster_theme: str
    confidence: float
    similar_keywords: List[str]
    sample_size: int
    processing_time_ms: float

class HealthResponse(BaseModel):

    status: str
    total_clusters: int
    uptime_seconds: float


vectorizer = None
kmeans_model = None

cluster_themes = {}
cluser_sizes = {}

feature_names = []
startup_time = time.time()

def load_models():

    global vectorizer, kmeans_model, cluster_themes, feature_names, cluser_sizes

    try:
        vectorizer = joblib.load("models/vectorizer.pkl")
        kmeans_model = joblib.load("models/kmeans_model.pkl")

        with open("models/cluster_themes.json", "r") as f:
            themes = json.load(f)

            cluster_themes = {int(k): v for k, v in themes.items()}

        feature_names = vectorizer.get_feature_names_out()

        for i in range(kmeans_model.n_clusters):
            cluser_sizes[i] = 1000

        print("Models loaded successfully.")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def preprocessor(text_vector, top_n: int = 5) -> List[str]:

    if hasattr(text_vector, "toarray"):
        vector_array = text_vector.toarray()

