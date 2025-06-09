from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import numpy as np
import faiss
import logging
import re
from typing import Optional

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MongoDB Setup ---
MONGO_URI = "mongodb+srv://Exotoura:ExotouraPWD@exotoura-cluster.mw1yq.mongodb.net/"
DB_NAME = "exotoura"
COLLECTION_NAME = "places"
mongo_client = MongoClient(MONGO_URI)
places_collection = mongo_client[DB_NAME][COLLECTION_NAME]

# --- Load FAISS Data ---
df = pd.read_pickle("./locations_data.pkl")
place_names = df['locationName'].tolist()
vectors = np.stack(df['combinedVector'].values)

# Normalize vectors for cosine similarity
norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
index = faiss.IndexFlatIP(norm_vectors.shape[1])
index.add(norm_vectors)

# --- FastAPI Setup ---
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

# --- Request & Response Models ---
class RecommendRequest(BaseModel):
    locationNames: Optional[list[str]] = None
    locationIds: Optional[list[str]] = None
    topN: int = 5

class PlaceResponse(BaseModel):
    locationName: str
    description: str
    latitude: float
    longitude: float
    rating: float
    reviewCount: int
    type: str
    imageURLs: list[str]
    approved: bool

class ScoredPlaceResponse(BaseModel):
    score: float
    data: PlaceResponse

class RecommendationResponse(BaseModel):
    topN: int
    results: list[ScoredPlaceResponse]

def dms_to_decimal(dms_str):
    dms_str = dms_str.strip()
    decimal_match = re.match(r"(\d+(\.\d+)?)\s*([NSEW])", dms_str)
    dms_match = re.match(r"(\d+)°(\d+)′(\d+)[″\"]?([NSEW])", dms_str)

    if decimal_match:
        value, _, direction = decimal_match.groups()
        decimal = float(value)
        if direction in ['S', 'W']:
            decimal *= -1
        return decimal

    if dms_match:
        deg, min_, sec, direction = dms_match.groups()
        decimal = int(deg) + int(min_) / 60 + int(sec) / 3600
        if direction in ['S', 'W']:
            decimal *= -1
        return decimal

    raise ValueError(f"Invalid coordinate format: {dms_str}")

def clean_document(doc):
    try:
        cleaned = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                cleaned[k] = str(v)
            elif k in ("latitude", "longitude") and isinstance(v, str):
                if any(d in v for d in "NSEW"):
                    cleaned[k] = dms_to_decimal(v)
                else:
                    cleaned[k] = float(v)
            else:
                cleaned[k] = v
        return cleaned
    except Exception as e:
        logger.warning(f"Failed to clean document: {e}")
        return None

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendRequest):
    query_names = []

    # Resolve from IDs
    if request.locationIds:
        object_ids = []
        for _id in request.locationIds:
            try:
                object_ids.append(ObjectId(_id))
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid ObjectId: {_id}")

        docs = list(places_collection.find({"_id": {"$in": object_ids}}, {"locationName": 1}))
        found_names = [doc["locationName"] for doc in docs if "locationName" in doc]
        if len(found_names) != len(object_ids):
            raise HTTPException(status_code=404, detail="Some place IDs not found.")
        query_names.extend(found_names)

    # Add direct names
    if request.locationNames:
        query_names.extend(request.locationNames)

    if not query_names:
        raise HTTPException(status_code=400, detail="Provide at least one of: locationNames or locationIds.")

    # Check names in FAISS
    invalid_names = [name for name in query_names if name not in place_names]
    if invalid_names:
        raise HTTPException(status_code=404, detail=f"Locations not found in FAISS index: {invalid_names}")

    indices = [place_names.index(name) for name in query_names]
    query_vectors = norm_vectors[indices]
    mean_vector = np.mean(query_vectors, axis=0, keepdims=True)

    D, I = index.search(mean_vector, request.topN + len(indices))

    found_names_scores = [
        (place_names[i], D[0][idx])
        for idx, i in enumerate(I[0])
        if place_names[i] not in query_names
    ][:request.topN]

    similar_places = [name for name, _ in found_names_scores]
    scores = [score for _, score in found_names_scores]

    logger.info(f"FAISS result for input {query_names}: {found_names_scores}")

    docs = list(places_collection.find({"locationName": {"$in": similar_places}}, {"_id": 0}))

    name_to_doc = {}
    for doc in docs:
        clean_doc = clean_document(doc)
        if clean_doc and 'locationName' in clean_doc:
            name_to_doc[clean_doc['locationName']] = clean_doc

    missing_places = [name for name in similar_places if name not in name_to_doc]
    if missing_places:
        raise HTTPException(status_code=500, detail=f"Missing documents in MongoDB: {missing_places}")

    scored_results = [
        {
            "score": round(float(score), 4),
            "data": name_to_doc[name]
        }
        for name, score in zip(similar_places, scores)
    ]

    return {"topN": len(scored_results), "results": scored_results}
