from flask import Flask, request, jsonify, send_from_directory
import os
import shutil
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Load YOLOv8 model for pole detection
model_pole = YOLO('poteau.pt')
# Load YOLOv8 model for lamp detection
model_lum = YOLO('best.pt')

def detect_poles(image_path):
    results = model_pole.predict(image_path, save=True, imgsz=640, conf=0.3, iou=0.5, augment=True, max_det=1)
    detections = []
    for result in results:
        for box in result.boxes:
            detection = {
                "name": result.names[box.cls[0].item()],
                "class": box.cls[0].item(),
                "confidence": box.conf[0].item(),
                "box": {
                    "x1": box.xyxy[0][0].item(),
                    "y1": box.xyxy[0][1].item(),
                    "x2": box.xyxy[0][2].item(),
                    "y2": box.xyxy[0][3].item()
                }
            }
            detections.append(detection)
    res_dir = results[0].save_dir if results else None
    return detections, res_dir

def detect_lum(image_path):
    results = model_lum.predict(image_path, save=True, imgsz=800, conf=0.4, iou=0.5, augment=True, max_det=10)
    detections = []
    for result in results:
        for box in result.boxes:
            detection = {
                "name": result.names[box.cls[0].item()],
                "class": box.cls[0].item(),
                "confidence": box.conf[0].item(),
                "box": {
                    "x1": box.xyxy[0][0].item(),
                    "y1": box.xyxy[0][1].item(),
                    "x2": box.xyxy[0][2].item(),
                    "y2": box.xyxy[0][3].item()
                }
            }
            detections.append(detection)
    res_dir = results[0].save_dir if results else None
    return detections, res_dir

@app.route('/api/detect', methods=['POST'])
def detect_objects_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    file.save(file_path)
    detections, _ = detect_poles(file_path)

    # Move the resulting image to a new folder
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_image.jpg')
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    shutil.move(file_path, result_image_path)
    return jsonify({'detections': detections})

@app.route('/api/detect1', methods=['POST'])
def detect_objects_api1():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    file.save(file_path)
    detections, _ = detect_lum(file_path)

    # Move the resulting image to a new folder
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_image.jpg')
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    shutil.move(file_path, result_image_path)
    return jsonify({'detections': detections})

@app.route('/api/detect_and_get_result_image', methods=['POST'])
def detect_and_get_result_image_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    file.save(temp_file_path)

    detections, _ = detect_poles(temp_file_path)

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)

    files = os.listdir(directory)
    latest_file = files[0]
    filename = os.path.join(directory, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file)
    else:
        return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/detect_and_get_result_image_lum', methods=['POST'])
def detect_and_get_result_image_lum_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    file.save(temp_file_path)

    detections, _ = detect_lum(temp_file_path)

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)

    files = os.listdir(directory)
    latest_file = files[0]
    filename = os.path.join(directory, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file)
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
