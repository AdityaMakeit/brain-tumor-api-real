# main.py
import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils import load_model, predict_image

# set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = load_model()

@app.get("/")
def read_root():
    return {"message": "Brain Tumor Classifier API 🧠"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file_path = f"/tmp/{file.filename}"
    try:
        # save upload
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # do prediction
        prediction = predict_image(temp_file_path, model)

        # cleanup
        os.remove(temp_file_path)

        return {"prediction": prediction}

    except Exception as e:
        # log full stack trace to server logs
        logger.exception("Prediction failed")
        # return something the client can see
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "details": str(e)}
        )
