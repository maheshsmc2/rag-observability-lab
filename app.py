# app.py

from fastapi import FastAPI
from pydantic import BaseModel

from retriever import answer_query  # 👈 import from retriever.py


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    """
    API endpoint:
    - receives JSON: {"query": "..."}
    - calls answer_query() from retriever.py
    - returns {"answer": "..."}
    """
    result = answer_query(req.query)
    return {"answer": result}



