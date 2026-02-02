from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to AutoScore!!"}

@app.get("/demo")
def home():
    return {"message": "Welcome to AutoScore!! Demo"}