from flask import Flask, request, jsonify
from keras.models import load_model
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np
import os
import torch
import cv2
import math


app = Flask(__name__)

# Load your Keras model
model = load_model("E:/Rem/nutritionapp/segmentationmodels/best_model.h5")
model_type = "DPT_Hybrid"  # choose model type
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Preprocess the image for the model
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def estimate_depth(image):
    img_py= np.array(image) 
    img_rgb = cv2.cvtColor(img_py, cv2.COLOR_RGB2BGR)
    input_batch = midas_transforms(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map


def calculate_mask_area(mask, ppi):
    pixel_area = np.count_nonzero(mask)
    
    # Convert pixel area to real-world area in square inches
    real_area_in_square_inches = (pixel_area / (ppi ** 2)) * (12 / 10) ** 2
    return real_area_in_square_inches

import numpy as np

def calculate_volume_and_weight(area_in_inches2, depth_map, ppi, density=0.96):
    """
    Calculate the volume and weight based on the area in inches², depth map, and PPI.
    - area_in_inches2: Total area of the object in inches² (pre-calculated).
    - depth_map: Depth map providing depth values for each pixel (in pixels).
    - ppi: Pixels per inch (PPI) for converting depth to real-world units.
    - density: Density of the object in g/cm³.
    """
    total_volume_in_inches3 = 0

    # Iterate through the depth map to accumulate the volume based on depth values
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            # Convert depth from pixels to inches
            depth_in_inches = depth_map[i, j] / ppi  # Depth in inches

            # Volume contribution for this pixel (depth in inches * area in inches²)
            pixel_volume_in_inches3 = depth_in_inches * (area_in_inches2 / depth_map.size)  # Approximate area per pixel
            total_volume_in_inches3 += pixel_volume_in_inches3

    # Convert volume from inches³ to cm³ (1 inch³ = 16.387 cm³)
    total_volume_in_cm3 = total_volume_in_inches3 * 16.387

    # Calculate the weight based on density (in grams)
    weight_in_grams = total_volume_in_cm3 * density  # Weight in grams (density in g/cm³)

    return total_volume_in_inches3



# Endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    ppi = 71
    # Preprocess and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    threshold = float(request.args.get("threshold", 0.012))
    
    predicted_mask = (prediction[0] > threshold).astype(np.uint8)
    # Convert mask to a list for JSON response (or use other formats as needed)

    area = calculate_mask_area(predicted_mask, ppi)

    depth_map = estimate_depth(image)

    mask_list = predicted_mask.tolist()
    

    volume = calculate_volume_and_weight(area, depth_map,ppi)


    return jsonify({
        "mask_area": volume
    })

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

'''food density
apple = 0.95 g/cm3 
chicken brest = 0.95 g/cm3 
steak = 0.92 g/cm3
spagetti = 0.769 g/cm³
bean = 0.80 g/cm3
boiled egg = 1.03 g/cm³
fried egg = 1.09 g/cm³'''