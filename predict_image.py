import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_path = 'cat_dog_classifier.keras'
image_size = (224, 224)

if not os.path.exists(model_path):
    print(f"Error: can't find model file '{model_path}'.Please check path or model。")
    exit()

print(f"Loading model：{model_path}...")
loaded_model = load_model(model_path)
print("Model loading successfully.")

def predict_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: can't find image file '{img_path}'.Please check path")
        return None, None

    try:
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = loaded_model.predict(img_array)[0][0]
        class_label = "Dog" if prediction > 0.5 else "Cat"
        confidence = prediction if prediction > 0.5 else (1 - prediction)

        print(f"Predict result：{class_label} (Confidence: {confidence:.4f})")
        return class_label, confidence

    except Exception as e:
        print(f"Error happens when handling images：{e}")
        return None, None
