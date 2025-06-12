
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from utils import preprocess_image
import numpy as np
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
model = tf.keras.models.load_model("Image_classify_Birds_Model.keras")

# Classes récupérées depuis data_cat de ton notebook
class_names = [
    'ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL',
    'AFRICAN CROWNED CRANE', 'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH',
    'AFRICAN OYSTER CATCHER', 'AFRICAN PIED HORNBILL', 'AFRICAN PYGMY GOOSE',
    'ALBATROSS', 'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET',
    'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET', 'AMERICAN BITTERN',
    'AMERICAN COOT', 'AMERICAN FLAMINGO', 'AMERICAN GOLDFINCH',
    'AMERICAN KESTREL'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    input_tensor = preprocess_image(image_data)

    # Prédiction brute
    logits = model.predict(input_tensor)

    # Softmax (comme dans le notebook)
    probs = tf.nn.softmax(logits[0])
    predicted_index = int(np.argmax(probs))
    predicted_class = class_names[predicted_index]
    confidence = float(probs[predicted_index])

    # Préparer chemin du fichier audio
    safe_filename = predicted_class + ".mp3"
    audio_path = f"/audio/{safe_filename}"

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "class_index": predicted_index,
        "audio_url": audio_path
    }
from fastapi.staticfiles import StaticFiles

app.mount("/audio", StaticFiles(directory="audio"), name="audio")
