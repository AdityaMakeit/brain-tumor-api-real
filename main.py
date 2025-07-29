# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from utils import load_model, predict_image

app = FastAPI()
model = load_model()

@app.get("/")
def read_root():
    return {"message": "Brain Tumor Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_image(temp_file_path, model)
    
    os.remove(temp_file_path)  # clean up
    return JSONResponse(content={"prediction": prediction})
