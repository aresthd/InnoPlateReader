from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from datetime import datetime
from utils import *

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.secret_key = "secret key"
app.config.update(
    UPLOADED_FOLDER_PREDICT = os.path.join('static', 'uploads'),
    UPLOADED_FOLDER_TRAIN_IMG = os.path.join('static', 'images', 'train'),
    UPLOADED_FOLDER_TRAIN_LABEL = os.path.join('static', 'labels', 'train'),
    DROPZONE_TIMEOUT = 5 * 60 * 1000,
    DROPZONE_ADD_REMOVE_LINKS = True
)
dropzone = Dropzone(app)

# UPLOAD_FOLDER = "\static\upload"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'txt'])
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CONFIG_PREDICT = {
    'filename' : None,
    'path_ori' : None,
    'path_ob' : None,
    'path_crop' : None,
    'path_td' : None
}

CONFIG_TRAIN = {
    'status' : None,
    'model_name' : None,
    'model' : None,
    'path_model' : None,
    'path_run' : None,
    'path_val' : None,
    'path_plot' : None,
    'accuracy' : None, 
    'precision' : None, 
    'recall' : None
}

MODEL_YOLO = "./model/best.pt"
CONFIG = "./static/config.yaml"


# Load a model
MODEL = YOLO(MODEL_YOLO)


@app.route("/")
@app.route("/index")
def index():
    data = {
        'page' : 'Home',
        'current_page' : 'home'
    }
    return render_template("pages/index.html", data=data)

@app.route("/about")
def about():
    data = {
        'page' : 'About',
        'current_page' : 'about'
    }
    return render_template("pages/about.html", data=data)


# @app.route("/login")
# def login():
#     return render_template("login.html")

# @app.route("/register")
# def register():
#     return render_template("register.html")

@app.route("/read-plate")
def read_plate():
    filepath = get_conf_predict(CONFIG_PREDICT,'path_ori')
    filename = get_conf_predict(CONFIG_PREDICT,'filename')
    data = {
        'page' : 'Read Plate',
        'current_page' : 'read plate'
    }
    if filepath != None:
        data['filepath'] = filepath
        data['filename'] = filename
    return render_template("pages/read_plate.html", data=data)

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        filename = secure_filename(f.filename)
        if f and allowed_file(filename, ALLOWED_EXTENSIONS):
            ext = get_file_extension(filename)
            if ext != "txt":
                filepath = os.path.join(app.config['UPLOADED_FOLDER_PREDICT'], filename)
                set_all_conf_predict(CONFIG_PREDICT, filename=filename, path_ori=filepath)
                f.save(filepath)
                return redirect("/read-plate")
            else:
                set_all_conf_predict(CONFIG_PREDICT, filename='error', path_ori='error')
                return redirect("/read-plate")
        else:
            set_all_conf_predict(CONFIG_PREDICT, filename='error', path_ori='error')
            return redirect("/read-plate")
    else:
        return redirect("/read-plate")

# Example Images
@app.route("/example-1", methods=['GET', 'POST'])
def example_1():
    filename = 'K2_AB4480UP.jpg'
    filepath = os.path.join('static', 'images', 'example', filename)
    set_conf_predict(CONFIG_PREDICT, 'filename', filename)
    set_conf_predict(CONFIG_PREDICT, 'path_ori', filepath)
    return redirect('/result')

@app.route("/example-2", methods=['GET', 'POST'])
def example_2():
    filename = 'K3_AB1510BA.jpg'
    filepath = os.path.join('static', 'images', 'example', filename)
    set_conf_predict(CONFIG_PREDICT, 'filename', filename)
    set_conf_predict(CONFIG_PREDICT, 'path_ori', filepath)
    return redirect('/result')

@app.route("/example-3", methods=['GET', 'POST'])
def example_3():
    filename = 'K4_H1753FO.jpg'
    filepath = os.path.join('static', 'images', 'example', filename)
    set_conf_predict(CONFIG_PREDICT, 'filename', filename)
    set_conf_predict(CONFIG_PREDICT, 'path_ori', filepath)
    return redirect('/result')
    
# Result Predict
@app.route("/result", methods=['GET', 'POST'])
def result():
    data = {
        'page' : 'Result Detect',
        'current_page' : 'read plate'
    }
    filepath, filename = get_conf_predict(CONFIG_PREDICT, 'path_ori'), get_conf_predict(CONFIG_PREDICT, 'filename')
    if filepath != None and filepath != 'error' and filename != 'error':
        result_text_plate = predict(MODEL, CONFIG_PREDICT, filepath, filename)
        pathOB, pathCrop, pathTD = get_conf_predict(CONFIG_PREDICT, 'path_ob'), get_conf_predict(CONFIG_PREDICT, 'path_crop'), get_conf_predict(CONFIG_PREDICT, 'path_td')
        data['filepath'], data['filename'], = filepath, filename
        data['pathOB'], data['pathCrop'], data['pathTD'] = pathOB, pathCrop, pathTD
        data['result_text_plate'] = result_text_plate
        return render_template('pages/result.html', data=data)
    elif filepath == 'error' and filename == 'error':
        set_all_conf_predict(CONFIG_PREDICT)
        flash('Please upload images in JPG or PNG format!', 'error')
        return redirect("/read-plate")
    else:
        return redirect('/read-plate')

@app.route("/download-result")
def download_result():
    filepath, filename = get_conf_predict(CONFIG_PREDICT, 'path_ori'), get_conf_predict(CONFIG_PREDICT, 'filename')
    if filepath != None:
        pathTD = get_conf_predict(CONFIG_PREDICT, 'path_td')
        filename = 'Result_' + filename
        return send_file(
            pathTD,
            download_name=filename,
            as_attachment=True,
        )   
    else:
        return redirect('/read-plate')
    
@app.route("/read-again")
def read_again():
    set_all_conf_predict(CONFIG_PREDICT)
    return redirect('/read-plate')

@app.route("/upgrade-model")
def upgrade_model():
    data = {
        'page' : 'Upgrade Model',
        'current_page' : 'upgrade model'
    }
    return render_template("pages/upgrade_model.html", data=data)
    
@app.route("/upload-train", methods=['GET', 'POST'])
def upload_train():
    if request.method == 'POST':
        f = request.files.get('file')
        filename = secure_filename(f.filename)
        if allowed_file(filename, ALLOWED_EXTENSIONS):
            ext = get_file_extension(filename)
            if ext == "txt":
                filepath = os.path.join(app.config['UPLOADED_FOLDER_TRAIN_LABEL'], filename)
                f.save(filepath)
            else:                
                filepath = os.path.join(app.config['UPLOADED_FOLDER_TRAIN_IMG'], filename)
                f.save(filepath)
            set_conf_train(CONFIG_TRAIN, 'status', 'start')
            return redirect("/upgrade-model")
        else: 
            set_conf_train(CONFIG_TRAIN, 'status', 'cancel')
            return redirect("/upgrade-model")
    else:
        return redirect("/upgrade-model")

@app.route("/train-model", methods=['GET', 'POST'])
def train_model():
    data = {
        'page' : 'Score Upgrade',
        'current_page' : 'upgrade model'
    }
    status = get_conf_train(CONFIG_TRAIN, 'status')
    if status != None and request.form['start-train'] == 'True':
        if status == 'start':
            epoch = int(request.form['epoch'])
            confidence = int(request.form['confidence'])
            model, accuracy, precision, recall = train(MODEL, CONFIG, CONFIG_TRAIN, epoch, confidence)
            status = get_conf_train(CONFIG_TRAIN, 'status')
            if status == 'success':
                plotTrain = get_conf_train(CONFIG_TRAIN, 'path_plot')
                set_conf_train(CONFIG_TRAIN, 'status', 'done')
                data['model'] = model
                data['accuracy'] = accuracy
                data['precision'] = precision
                data['recall'] = recall
                data['plotTrain'] = plotTrain
                return render_template('pages/score.html', data=data)
            elif status == 'no image':
                flash('Please upload the images too!', 'error')
            elif status == 'no label':
                flash('Please upload the labels too!', 'error')
            elif status == 'add data':
                flash('Please upload more data!', 'error')
            return redirect("/upgrade-model")
        elif status == 'done':
            flash('Please upload the images and labels again!', 'error')
            return redirect("/upgrade-model")
        else:
            flash('Please upload the images (png or jpg) and labels (txt)!', 'error')
            return redirect("/upgrade-model")
    else:
        flash('Please upload the images and labels!', 'error')
        return redirect('/upgrade-model')

@app.route("/save-model")
def save_model():
    modelName, pathModel = get_conf_train(CONFIG_TRAIN, 'model_name'), get_conf_train(CONFIG_TRAIN, 'path_model')
    if pathModel != None:
        modelName = 'Model_' + modelName + '.pt'
        return send_file(
            pathModel,
            download_name=modelName,
            as_attachment=True,
        )   
    else:
        return redirect('/score')

@app.route("/score")
def score():
    data = {
        'page' : 'Score Upgrade',
        'current_page' : 'upgrade model'
    }
    return render_template('pages/score.html', data)

# @app.route("/res")
# def res():
#     return render_template('result.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)