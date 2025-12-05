from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AutoScore FastAPI running!"}
@app.get("/hello")
def root3():
    return {"message": "AutoScore FastAPI running!"}
