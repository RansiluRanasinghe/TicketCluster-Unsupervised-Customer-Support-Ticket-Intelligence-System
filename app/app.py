from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import time
import os
from typing import List, Dict

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
cluster_sizes = {}
feature_names = []
startup_time = time.time()

def load_models():
    global vectorizer, kmeans_model, cluster_themes, feature_names, cluster_sizes
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        vectorizer = joblib.load(os.path.join(base_dir, "app/models", "vectorizer.pkl"))
        kmeans_model = joblib.load(os.path.join(base_dir, "app/models", "kmeans_model.pkl"))
        
        with open(os.path.join(base_dir, "app/models", "cluster_themes.json"), "r") as f:
            themes = json.load(f)
            cluster_themes = {int(k): v for k, v in themes.items()}
        
        feature_names = vectorizer.get_feature_names_out()
        
        for i in range(kmeans_model.n_clusters):
            cluster_sizes[i] = 1000
        
        print("Models loaded successfully.")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def preprocessor(subject: str, body: str) -> str:
    combined = f"{subject} {body}".lower().strip()
    import re
    combined = re.sub(r'\s+', ' ', combined)
    return combined

def extract_keywords(text_vector, top_n: int = 5) -> List[str]:
    if hasattr(text_vector, "toarray"):
        vector_array = text_vector.toarray().flatten()
    else:
        vector_array = text_vector.flatten()
    
    nonzero = np.where(vector_array > 0)[0]
    
    if len(nonzero) == 0:
        return []
    
    top_indices = nonzero[np.argsort(vector_array[nonzero])[-top_n:][::-1]]
    return [feature_names[i] for i in top_indices]

def calc_confidence(text_vector, cluster_id: int) -> float:
    centroid = kmeans_model.cluster_centers_[cluster_id]
    
    if hasattr(text_vector, "toarray"):
        vector_array = text_vector.toarray().flatten()
    else:
        vector_array = text_vector.flatten()
    
    try:
        dot_product = np.dot(vector_array, centroid)
        norm_vector = np.linalg.norm(vector_array)
        norm_centroid = np.linalg.norm(centroid)
        
        if norm_vector == 0 or norm_centroid == 0:
            return 0.5
        
        similarity = dot_product / (norm_vector * norm_centroid)
        return max(0.0, min(1.0, similarity))
    except:
        return 0.5

app = FastAPI(
    title="Ticket Cluster API",
    description="An API for clustering customer support tickets using unsupervised learning techniques.",
    version= "1.0.0"
)

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/")
def root():
     return {
        "message": "Welcome to TicketCluster API",
        "endpoints": {
            "health": "/health",
            "cluster": "POST /cluster",
            "cluster_info": "/clusters",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    if vectorizer is None or kmeans_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    return HealthResponse(
        status="healthy",
        total_clusters=kmeans_model.n_clusters,
        uptime_seconds=time.time() - startup_time
    )

@app.post("/cluster", response_model=ClusterResponse)
def cluster_tickets(ticket: Ticket):
    if vectorizer is None or kmeans_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    start_time = time.time()

    try:
        processed_text = preprocessor(ticket.subject, ticket.body)
        text_vector = vectorizer.transform([processed_text])

        cluster_id = int(kmeans_model.predict(text_vector)[0])

        confidence = calc_confidence(text_vector, cluster_id)

        keywords = extract_keywords(text_vector, top_n=5)
        theme = cluster_themes.get(cluster_id, "General Support")

        sample_size = cluster_sizes.get(cluster_id, 0)

        processing_time = (time.time() - start_time) * 1000

        return ClusterResponse(
            cluster_id=cluster_id,
            cluster_theme=theme,
            confidence=confidence,
            similar_keywords=keywords,
            sample_size=sample_size,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/cluster/batch")
def cluster_batch(tickets: List[Ticket]):
    if vectorizer is None or kmeans_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    start_time = time.time()

    results = []

    for ticket in tickets:
        try:
            processed_text = preprocessor(ticket.subject, ticket.body)
            text_vector = vectorizer.transform([processed_text])
            cluster_id = int(kmeans_model.predict(text_vector)[0])
            confidence = calc_confidence(text_vector, cluster_id)
            keywords = extract_keywords(text_vector, top_n=5)
            theme = cluster_themes.get(cluster_id, "General Support")
            
            results.append({
                "subject": ticket.subject[:50] + "..." if len(ticket.subject) > 50 else ticket.subject,
                "cluster_id": cluster_id,
                "cluster_theme": theme,
                "confidence": confidence,
                "keywords": keywords
            })

        except Exception as e:
            results.append({
                "subject": ticket.subject[:50] + "..." if len(ticket.subject) > 50 else ticket.subject,
                "error": str(e)
            })

    total_time = (time.time() - start_time) * 1000

    return {
        "total_tickets": len(tickets),
        "processed_tickets": len([r for r in results if "error" not in r]),
        "total_time_ms": total_time,
        "results": results
    }
    
@app.get("/clusters")
def get_clusters():
    if kmeans_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    clusters_info = []
    
    for cluster_id in range(kmeans_model.n_clusters):
        theme = cluster_themes.get(cluster_id, "Unknown")

        centroid = kmeans_model.cluster_centers_[cluster_id]
        top_indices = np.argsort(centroid)[-10:][::-1]
        top_keywords = [feature_names[i] for i in top_indices[:5]]
        
        clusters_info.append({
            "cluster_id": cluster_id,
            "theme": theme,
            "size": cluster_sizes.get(cluster_id, 0),
            "top_keywords": top_keywords
        })
    
    return {
        "total_clusters": kmeans_model.n_clusters,
        "features": len(feature_names),
        "clusters": clusters_info
    }

@app.get("/clusters/{cluster_id}")
def get_cluster_info(cluster_id: int):
    if kmeans_model is None or cluster_id >= kmeans_model.n_clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    theme = cluster_themes.get(cluster_id, "Unknown")
    
    centroid = kmeans_model.cluster_centers_[cluster_id]
    top_indices = np.argsort(centroid)[-15:][::-1]
    top_keywords = [{"word": feature_names[i], "score": float(centroid[i])} 
                    for i in top_indices[:10]]
    
    return {
        "cluster_id": cluster_id,
        "theme": theme,
        "size": cluster_sizes.get(cluster_id, 0),
        "centroid_keywords": top_keywords,
        "confidence_threshold": 0.5 
    }
        
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )