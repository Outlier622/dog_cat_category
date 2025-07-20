import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# 模型路径（你训练时保存的 .keras 模型）
MODEL_PATH = 'cat_breed_classifier_finetuned_6.keras'
CLASS_INDEX_PATH = 'cat_class_indices.npy'  # 训练时保存的 class_indices 映射

# 加载模型
print("加载猫品种模型：", MODEL_PATH)
model = load_model(MODEL_PATH)

# 加载类别映射字典（如 {0: 'Abyssinian', 1: 'Persian', ...}）
if not os.path.exists(CLASS_INDEX_PATH):
    raise FileNotFoundError(f"未找到类别映射文件：{CLASS_INDEX_PATH}")
class_indices = np.load(CLASS_INDEX_PATH, allow_pickle=True).item()
index_to_class = {v: k for k, v in class_indices.items()}

# 图像预测函数
def predict_cat_breed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    top_index = np.argmax(prediction)
    breed_name = index_to_class[top_index]
    confidence = prediction[top_index]

    return breed_name, float(confidence)
