import os
import cv2

input_folder = r"D:\FYP Datasets\BAGLS\training\training"
output_folder = r"D:\FYP Datasets\BAGLS\training_bb"

def detect_bounding_boxes(img):
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    cntrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    #convert to yolo
    x_center = (x_min + x_max) / 2 / img.shape[1]
    y_center = (y_min + y_max) / 2 / img.shape[0]
    width = (x_max - x_min) / img.shape[1]
    height = (y_max - y_min) / img.shape[0]
    return x_center, y_center, width, height

for file in os.listdir(input_folder):
    if file.endswith("_seg.png"):
        img = cv2.imread(os.path.join(input_folder, file), cv2.IMREAD_GRAYSCALE)
        x, y, w, h = detect_bounding_boxes(img)
        if x == float('inf') or y == float('inf'):
            continue
        #write in a new text file and save in another folder
        with open(os.path.join(output_folder, file.replace("_seg.png", ".txt")), "w") as f: #replace _seg.png with .txt to follow 
            f.write(f"3 {x} {y} {w} {h}")
            