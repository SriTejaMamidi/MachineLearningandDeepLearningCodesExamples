from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"message":"Hello World"}
@app.get("/predict")
def home():
    return {"prediction":"positive"}