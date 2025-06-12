"""from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)"""
"""from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # comme Rescaling(1./255)
    return np.expand_dims(image_array, axis=0)  # format (1, 224, 224, 3)
"""
import tensorflow as tf
import io

def preprocess_image(image_bytes, target_size=(224, 224)):
    # Charger l'image depuis les bytes
    image = tf.keras.utils.load_img(io.BytesIO(image_bytes), target_size=target_size)
    image_tensor = tf.expand_dims(image, axis=0)  # Ajouter batch dimension
    return image_tensor

