import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os # 用于文件路径操作

# --- 定义预测所需参数 ---
model_path = 'best_mobilenetv2_model.h5' # 确保这个路径正确指向你保存的模型文件
image_size = (224, 224)              # 模型的输入图片尺寸，必须与训练时相同

# 确认模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误：找不到模型文件 '{model_path}'。请检查路径或确保模型已成功保存。")
    exit()

# --- 加载模型 ---
print(f"加载模型：{model_path}...")
loaded_model = load_model(model_path)
print("模型加载成功。")

# --- 定义预测函数 ---
def predict_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"错误：找不到图片文件 '{img_path}'。请检查路径。")
        return None, None

    print(f"处理图片：{img_path}...")
    try:
        # 加载图片并调整大小
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img) # 将图片转换为 NumPy 数组

        # 扩展维度以匹配模型输入格式 (批次维度)
        # 模型期望输入形状为 (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # 像素归一化，与训练时保持一致
        img_array /= 255.0

        # 进行预测
        prediction = loaded_model.predict(img_array)[0][0] # 对于二分类，sigmoid 输出一个值

        # 根据 sigmoid 输出值判断类别
        # 在 `cdmodel.py` 中，如果 `cat` 是 0，`dog` 是 1，那么：
        # 如果 prediction > 0.5，更可能是 dog
        # 如果 prediction <= 0.5，更可能是 cat
        # 请根据你在训练时 `train_generator.class_indices` 的输出确认猫狗的索引。
        # 通常 {'cat': 0, 'dog': 1}
        
        class_label = "Dog" if prediction > 0.5 else "Cat"
        confidence = prediction if prediction > 0.5 else (1 - prediction) # 获取较高置信度

        print(f"预测结果：{class_label} (置信度: {confidence:.4f})")
        return class_label, confidence

    except Exception as e:
        print(f"处理图片时发生错误：{e}")
        return None, None

# --- 示例：使用一张新图片进行预测 ---
# 请将 'path/to/your/new_image.jpg' 替换为你要预测的图片路径！
# 最好是你的训练和验证集中没有的全新图片
test_image_path = 'animals/cat/00000-4122619873.png'

# 假设你在项目根目录有一个名为 'test_images' 的文件夹，里面放了测试图片
# 例如：test_image_path = os.path.join('test_images', 'some_new_cat.jpg')

# 运行预测
predicted_label, predicted_confidence = predict_single_image(test_image_path)

if predicted_label:
    print(f"\n最终预测：图片是 {predicted_label}，置信度为 {predicted_confidence:.2%}")