import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


print("Loading model：dog_breed_classifier.keras...")
model = load_model("dog_breed_classifier.keras")
print("Model loading successfully")


class_indices = {v: k for k, v in np.load("class_indices.npy", allow_pickle=True).item().items()}

def predict_dog_breed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    class_idx = np.argmax(preds)
    breed = class_indices[class_idx]
    confidence = float(preds[class_idx])
    return breed, confidence
