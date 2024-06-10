from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import shutil
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Load YOLOv8 model for pole detection
model_pole = YOLO('poteau.pt')
# Load YOLOv8 model for lamp detection
model_lum =YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html', uploaded_image=None)

###############################################################################################################
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    #Cette condition vérifie si la méthode de la requête actuelle est POST. Cela signifie que le formulaire a été soumis et que les données du formulaire sont envoyées via la méthode POST.
    if request.method == 'POST':
        #Cette condition vérifie si le fichier est présent dans les fichiers de la requête. Si ce n'est pas le cas, cela signifie que l'utilisateur n'a pas sélectionné de fichier à télécharger.
        if 'file' not in request.files:
            #Si aucun fichier n'est présent, cette ligne redirige l'utilisateur vers la même page de téléchargement pour lui permettre de sélectionner un fichier.
            return redirect(request.url)
        #Cette ligne récupère le fichier téléchargé à partir des fichiers de la requête. 'file' est le nom de l'élément de formulaire HTML qui contient le fichier téléchargé.
        file = request.files['file']
        #Cette condition vérifie si le nom du fichier est vide. Cela peut se produire si l'utilisateur a soumis le formulaire sans sélectionner de fichier.
        if file.filename == '':
            return redirect(request.url)
        #Cette condition vérifie si un fichier a été téléchargé avec succès. Si c'est le cas, le bloc suivant de code est exécuté.
        if file:
            #Cette ligne crée le chemin complet du fichier téléchargé sur le serveur. Il utilise la configuration de l'application pour obtenir le dossier de téléchargement et crée un chemin vers un fichier nommé 'uploaded_image.jpg' dans ce dossier.
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
            file.save(file_path)
            detection_data, _ = detect_poles(file_path)
            return redirect(url_for('display', filename='uploaded_image.jpg'))
    return render_template('upload.html')
######################################################################################################
def detect_poles(image_path):
    # Run inference on the source
    results = model_pole.predict(image_path, save=True, imgsz=640, conf=0.3, iou=0.5, augment=True, max_det=1)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        res_dir = result.save_dir

    return results[0].tojson(), res_dir

#####################################################################################################

# Function for another detection task
def detect_lum(image_path):
    results = model_lum.predict(image_path, save=True, imgsz=800, conf=0.4, iou=0.5, augment=True, max_det=10)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        res_dir = result.save_dir

    return results[0].tojson(), res_dir

#######################################################################################################
@app.route('/display/<path:filename>')
def display(filename):
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
        return "Invalid file format"

####################################################################################################
@app.route('/api/detect', methods=['POST'])
def detect_objects_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    file.save(file_path)
    results, _ = detect_poles(file_path)

    # Move the resulting image to a new folder
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_image.jpg')
    # Create the result_image.jpg folder if it does not exist
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    shutil.move(file_path, result_image_path)
    return jsonify({'detections': results})

############################################################################################

@app.route('/upload_lum', methods=['POST', 'GET'])
def upload_lum():
    #Cette condition vérifie si la méthode de la requête actuelle est POST. Cela signifie que le formulaire a été soumis et que les données du formulaire sont envoyées via la méthode POST.
    if request.method == 'POST':
        #Cette condition vérifie si le fichier est présent dans les fichiers de la requête. Si ce n'est pas le cas, cela signifie que l'utilisateur n'a pas sélectionné de fichier à télécharger.
        if 'file' not in request.files:
            #Si aucun fichier n'est présent, cette ligne redirige l'utilisateur vers la même page de téléchargement pour lui permettre de sélectionner un fichier.
            return redirect(request.url)
        #Cette ligne récupère le fichier téléchargé à partir des fichiers de la requête. 'file' est le nom de l'élément de formulaire HTML qui contient le fichier téléchargé.
        file = request.files['file']
        #Cette condition vérifie si le nom du fichier est vide. Cela peut se produire si l'utilisateur a soumis le formulaire sans sélectionner de fichier.
        if file.filename == '':
            return redirect(request.url)
        #Cette condition vérifie si un fichier a été téléchargé avec succès. Si c'est le cas, le bloc suivant de code est exécuté.
        if file:
            #Cette ligne crée le chemin complet du fichier téléchargé sur le serveur. Il utilise la configuration de l'application pour obtenir le dossier de téléchargement et crée un chemin vers un fichier nommé 'uploaded_image.jpg' dans ce dossier.
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image_lamp.jpg')
            file.save(file_path)
            detection_data, _ = detect_lum(file_path)
            return redirect(url_for('display', filename='uploaded_image_lamp.jpg'))
    return render_template('upload1.html')
##################################################################################
@app.route('/api/detect1', methods=['POST'])
def detect_objects_api1():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    file.save(file_path)
    results, _ = detect_lum(file_path)

    # Move the resulting image to a new folder
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_image.jpg')
    # Create the result_image.jpg folder if it does not exist
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    shutil.move(file_path, result_image_path)
    return jsonify({'detections': results})

########################################################################
@app.route('/api/detect_and_get_result_image', methods=['POST'])
def detect_and_get_result_image_api():
    # Vérifier si le fichier est dans la requête
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Sauvegarder le fichier dans un dossier temporaire
    file = request.files['file']
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    file.save(temp_file_path)

    # Exécuter la détection sur le fichier
    results, _ = detect_poles(temp_file_path)

    # Récupérer le dossier où les détections sont enregistrées
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)

    # Récupérer le fichier d'image résultant
    files = os.listdir(directory)
    latest_file = files[0]
    filename = os.path.join(directory, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()

    # Vérifier si le fichier est une image au format JPG
    if file_extension == 'jpg':
        # Envoyer le fichier d'image résultant avec les détections
        return send_from_directory(directory, latest_file)
    else:
        return jsonify({'error': 'Invalid file format'}), 400
#################################################################################################

@app.route('/api/detect_and_get_result_image_lum', methods=['POST'])
def detect_and_get_result_image_lum_api():
    # Vérifier si le fichier est dans la requête
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Sauvegarder le fichier dans un dossier temporaire
    file = request.files['file']
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
    file.save(temp_file_path)

    # Exécuter la détection sur le fichier
    results, _ = detect_lum(temp_file_path)

    # Récupérer le dossier où les détections sont enregistrées
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)

    # Récupérer le fichier d'image résultant
    files = os.listdir(directory)
    latest_file = files[0]
    filename = os.path.join(directory, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()

    # Vérifier si le fichier est une image au format JPG
    if file_extension == 'jpg':
        # Envoyer le fichier d'image résultant avec les détections
        return send_from_directory(directory, latest_file)
    else:
        return jsonify({'error': 'Invalid file format'}), 400
    ###############################################################################

@app.route('/display_with_json/<path:filename>')
def display_with_json(filename):
    # Récupérer les données de détection associées à l'image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if 'lamp' in filename:
        detection_data, _ = detect_lum(file_path)
    else:
        detection_data, _ = detect_poles(file_path)


    # Charger le modèle pour le rendu de la page HTML
    return render_template('display_with_json.html', filename=filename, detection_data=detection_data)

if __name__ == '__main__':
    app.run(debug=True)
