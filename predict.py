from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

class_names = [
    'long sleeve dress',
    'long sleeve top',
    'short sleeve dress',
    'short sleeve top',
    'shorts',
    'skirt',
    'trousers',
    'vest',
    'vest dress'
]

model_path = "my_model.keras"
if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/1GW03Q8CkjcycjJ38V6wo-4oaGjOOph8_/view?usp=drive_link"  # Replace with your real file ID
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence
