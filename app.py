from fastapi import FastAPI

app=FastAPI()

app.post("/score")
def score():
    return {"score": 42}  # Placeholder implementation
