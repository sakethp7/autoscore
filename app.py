from fastapi import FastAPI

app=FastAPI()

app.get("/score")
def score():
    return {"score": 42}  # Placeholder implementation
