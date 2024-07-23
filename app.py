from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
import os
import cv2
import imutils
import easyocr
from ultralytics import YOLO
from datetime import datetime


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
model = YOLO(MODEL_YOLO)

# Train
def train(epoch, confidence):
    global model
    basedirTraining = os.path.join('static', 'training')
    confidence = confidence / 100

    date = f'{str(datetime.now().day)}_{str(datetime.now().month)}_{str(datetime.now().year)}'
    time = f'{str(datetime.now().hour)}_{str(datetime.now().minute)}_{str(datetime.now().second)}'
    name = f'{date}_{time}'
    
    try:
        result = model.train(data=CONFIG, epochs=epoch, project=basedirTraining, name=name)
        print(result)
        metrics = model.val(conf=confidence)
        # pathPlot = "static/dist/img"
        # metrics.confusion_matrix.plot(normalize=False, save_dir=pathPlot)
        
        pathRun = result.save_dir
        pathVal = metrics.save_dir
        pathPlot = os.path.join(pathRun, "confusion_matrix.png")
        print(f'\n\n pathRun: {pathRun}\n')
        print(f'\n\n pathPlot: {pathPlot}\n')
        pathModel = os.path.join(pathRun, 'weights', 'best.pt')
        
        cf = metrics.confusion_matrix.matrix
        tp = cf[0][0]
        fp = cf[0][1]
        fn = cf[1][0]
        tn = cf[1][1]
        accuracy = ((tp + tn) / (tp+fp+fn+tn)) * 100
        precision = (tp / (tp+fp)) * 100
        recall = (tp /(tp+fn)) * 100
        
        set_all_conf_train('success', name, model, pathModel, pathRun, pathVal, pathPlot, accuracy, precision, recall)
        return model, accuracy, precision, recall
    except IndexError as ae:
        accuracy = 0
        precision = 0
        recall = 0
        set_conf_train('status', 'add data')
        return model, accuracy, precision, recall
    except FileNotFoundError as ae:
        accuracy = 0
        precision = 0
        recall = 0
        set_conf_train('status', 'no image')
        return model, accuracy, precision, recall
    except AttributeError as ae:
        accuracy = 0
        precision = 0
        recall = 0
        set_conf_train('status', 'no label')
        return model, accuracy, precision, recall
    
# Predict
def predict(path_image, filename):
    basedirPredict = 'static\images\predict'
    
    # Load image
    img = cv2.imread(path_image)
    imgSize = img.shape[0:2]

    # Predict plat
    results = model(path_image, conf=0.1)
    results = results[0]
    x1, y1, x2, y2 = None, None, None, None
    
    # Draw rectangle
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, classId = result
        croppedImage = img[int(y1):int(y2), int(x1):int(x2)]
        # print(results.names[int(classId)].upper())
    
    # Simpan hasil crop
    pathImgCrop = os.path.join(basedirPredict, 'Crop_' + filename)
    cv2.imwrite(pathImgCrop, croppedImage)
    
    # Melakukan text recognition dengan easy ocr
    reader = easyocr.Reader(['en'])
    result = reader.readtext(croppedImage)
    
    # Define Font
    font = cv2.FONT_HERSHEY_SIMPLEX     # Jenis font
    fontScale = 3                      # Skala font
    fontThickness = 5                  # Ketebalan font
    fontColor = (0, 255, 0)            # Warna biru (BGR)
    
    # Mengambil text hasil deteksi OCR
    textToWrite = " ".join([res[1] for res in result])

    # Gambar rectangle box
    imgDrawBox = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), fontColor, fontThickness)  # fontColor adalah warna merah
    pathImgOb = os.path.join(basedirPredict, 'OB_' + filename)
    cv2.imwrite(pathImgOb, imgDrawBox)
    
    # Tulis kata
    imgDrawText = cv2.putText(imgDrawBox, textToWrite, (int(x1), int(y1) - 10), font, fontScale, fontColor, fontThickness)  # fontColor adalah warna merah
    pathImgTD = os.path.join(basedirPredict, 'TD_' + filename)
    cv2.imwrite(pathImgTD, imgDrawText)
    
    set_all_conf_predict(filename, path_image, pathImgOb, pathImgCrop, pathImgTD)
    return textToWrite

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(file_name):
    try:
        file_extension = file_name.split('.')[-1]
        return file_extension.lower() 
    except IndexError:
        return None

def set_all_conf_predict(filename=None, path_ori=None, path_ob=None, path_crop=None, path_td=None):
    global CONFIG_PREDICT
    CONFIG_PREDICT['filename'] = filename
    CONFIG_PREDICT['path_ori'] = path_ori
    CONFIG_PREDICT['path_ob'] = path_ob
    CONFIG_PREDICT['path_crop'] = path_crop
    CONFIG_PREDICT['path_td'] = path_td

def set_all_conf_train(status=None, model_name=None, model=None, path_model=None, path_run=None, path_val=None, path_plot=None, accuracy=None,  precision=None,  recall=None):
    global CONFIG_TRAIN
    CONFIG_TRAIN['status'] = status
    CONFIG_TRAIN['model_name'] = model_name
    CONFIG_TRAIN['model'] = model
    CONFIG_TRAIN['path_model'] = path_model
    CONFIG_TRAIN['path_run'] = path_run
    CONFIG_TRAIN['path_val'] = path_val
    CONFIG_TRAIN['path_plot'] = path_plot
    CONFIG_TRAIN['accuracy' ] = accuracy
    CONFIG_TRAIN['precision' ] = precision
    CONFIG_TRAIN['recall'] = recall

def set_conf_predict(keyword, value):
    global CONFIG_PREDICT
    CONFIG_PREDICT[keyword] = value

def set_conf_train(keyword, value):
    global CONFIG_TRAIN
    CONFIG_TRAIN[keyword] = value

def get_conf_predict(keyword):
    global CONFIG_PREDICT
    return CONFIG_PREDICT[keyword]

def get_conf_train(keyword):
    global CONFIG_TRAIN
    return CONFIG_TRAIN[keyword]


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
    filepath = get_conf_predict('path_ori')
    filename = get_conf_predict('filename')
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
        if f and allowed_file(filename):
            ext = get_file_extension(filename)
            if ext != "txt":
                filepath = os.path.join(app.config['UPLOADED_FOLDER_PREDICT'], filename)
                set_all_conf_predict(filename=filename, path_ori=filepath)
                f.save(filepath)
                return redirect("/read-plate")
            else:
                set_all_conf_predict(filename='error', path_ori='error')
                return redirect("/read-plate")
        else:
            set_all_conf_predict(filename='error', path_ori='error')
            return redirect("/read-plate")
    else:
        return redirect("/read-plate")

# Example Images
@app.route("/example-1", methods=['GET', 'POST'])
def example_1():
    filename = 'K2_AB4480UP.jpg'
    filepath = os.path.join('static', 'images', 'example', filename)
    set_conf_predict('filename', filename)
    set_conf_predict('path_ori', filepath)
    return redirect('/result')

@app.route("/example-2", methods=['GET', 'POST'])
def example_2():
    filename = 'K3_AB1510BA.jpg'
    filepath = os.path.join('static', 'images', 'example', filename)
    set_conf_predict('filename', filename)
    set_conf_predict('path_ori', filepath)
    return redirect('/result')

@app.route("/example-3", methods=['GET', 'POST'])
def example_3():
    filename = 'K4_H1753FO.jpg'
    filepath = os.path.join('static', 'images', 'example', filename)
    set_conf_predict('filename', filename)
    set_conf_predict('path_ori', filepath)
    return redirect('/result')
    
# Result Predict
@app.route("/result", methods=['GET', 'POST'])
def result():
    data = {
        'page' : 'Result Detect',
        'current_page' : 'read plate'
    }
    filepath, filename = get_conf_predict('path_ori'), get_conf_predict('filename')
    if filepath != None and filepath != 'error' and filename != 'error':
        result_text_plate = predict(filepath, filename)
        pathOB, pathCrop, pathTD = get_conf_predict('path_ob'), get_conf_predict('path_crop'), get_conf_predict('path_td')
        data['filepath'], data['filename'], = filepath, filename
        data['pathOB'], data['pathCrop'], data['pathTD'] = pathOB, pathCrop, pathTD
        data['result_text_plate'] = result_text_plate
        return render_template('pages/result.html', data=data)
    elif filepath == 'error' and filename == 'error':
        set_all_conf_predict()
        flash('Please upload images in JPG or PNG format!', 'error')
        return redirect("/read-plate")
    else:
        return redirect('/read-plate')

@app.route("/download-result")
def download_result():
    filepath, filename = get_conf_predict('path_ori'), get_conf_predict('filename')
    if filepath != None:
        pathTD = get_conf_predict('path_td')
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
    set_all_conf_predict()
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
        if allowed_file(filename):
            print('\n Mengecek ekstensi file...\n')
            ext = get_file_extension(filename)
            if ext == "txt":
                print(f'\n File txt...\n')
                print(f'\n Mulai menyimpan file {ext}...\n')
                filepath = os.path.join(app.config['UPLOADED_FOLDER_TRAIN_LABEL'], filename)
                f.save(filepath)
            else:                
                print(f'\n File bukan txt...\n')
                print(f'\n Mulai menyimpan file {ext}...\n')
                filepath = os.path.join(app.config['UPLOADED_FOLDER_TRAIN_IMG'], filename)
                f.save(filepath)
            print('\n File berhasil disimpan...\n')
            set_conf_train('status', 'start')
            print('\n--------------')
            print(filepath)
            print(filename)
            print('\n\n')
            return redirect("/upgrade-model")
        else: 
            print('\n File sembarangan...\n')
            set_conf_train('status', 'cancel')
            return redirect("/upgrade-model")
    else:
        return redirect("/upgrade-model")

@app.route("/train-model", methods=['GET', 'POST'])
def train_model():
    data = {
        'page' : 'Score Upgrade',
        'current_page' : 'upgrade model'
    }
    print('\n--------------')
    print('\n\n Mengakses /train....\n')
    status = get_conf_train('status')
    print(f'\n Status : {status}....\n')
    if status != None and request.form['start-train'] == 'True':
        if status == 'start':
            epoch = int(request.form['epoch'])
            confidence = int(request.form['confidence'])
            print('\n\n Start Train Data....\n')
            model, accuracy, precision, recall = train(epoch, confidence)
            status = get_conf_train('status')
            if status == 'success':
                plotTrain = get_conf_train('path_plot')
                set_conf_train('status', 'done')
                data['model'] = model
                data['accuracy'] = accuracy
                data['precision'] = precision
                data['recall'] = recall
                data['plotTrain'] = plotTrain
                return render_template('pages/score.html', data=data)
            elif status == 'no image':
                flash('Please upload the images too!', 'error')
                return redirect("/upgrade-model")
            elif status == 'no label':
                flash('Please upload the labels too!', 'error')
                return redirect("/upgrade-model")
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
        print('\n\n Mengembalikkan ke halaman upgrade model....\n')
        flash('Please upload the images and labels!', 'error')
        return redirect('/upgrade-model')

@app.route("/save-model")
def save_model():
    print('\n--------------')
    print('\n\n Mengakses /save-model....\n')
    modelName, pathModel = get_conf_train('model_name'), get_conf_train('path_model')
    if pathModel != None:
        modelName = 'Model_' + modelName + '.pt'
        return send_file(
            pathModel,
            download_name=modelName,
            as_attachment=True,
        )   
    else:
        print('\n\n Mengembalikkan ke halaman upgrade model....\n')
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