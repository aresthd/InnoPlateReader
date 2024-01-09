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

FILEPATH_PREDICT = {
    'filename' : None,
    'original' : None,
    'object_detection' : None,
    'crop_object' : None,
    'text_detection' : None
}

FILEPATH_TRAIN = {
    'status' : None,
    'result' : None,
    'val' : None,
    'plot' : None,
    'model' : None
}

RESULT_TRAIN = {
    'model_name' : None,
    'model' : None,
    'accuracy' : None, 
    'precision' : None, 
    'recall' : None
}

MODEL_YOLO = "D:/Coding/Projects/number-plate-recognition/model/best.pt"
CONFIG = "D:/Coding/Projects/number-plate-recognition/static/config.yaml"


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
    
    result = model.train(data=CONFIG, epochs=epoch, project=basedirTraining, name=name)
    metrics = model.val(conf=confidence)
    
    pathPlot = "static/dist/img"
    metrics.confusion_matrix.plot(normalize=False, save_dir=pathPlot)
    
    pathResult = result.save_dir
    pathVal = metrics.save_dir
    pathPlot = os.path.join(pathPlot, "confusion_matrix.png")
    pathModel = os.path.join(pathResult, 'weights', 'best.pt')
    set_path_train(pathResult, pathVal, pathPlot, pathModel)
    
    cf = metrics.confusion_matrix.matrix
    tp = cf[0][0]
    fp = cf[0][1]
    fn = cf[1][0]
    tn = cf[1][1]
    accuracy = ((tp + tn) / (tp+fp+fn+tn)) * 100
    precision = (tp / (tp+fp)) * 100
    recall = (tp /(tp+fn)) * 100
    set_res_train(name, model, accuracy, precision, recall)
    
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
    
    set_path_predict_result(pathImgOb, pathImgCrop, pathImgTD)
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

def set_path_none():
    global FILEPATH_PREDICT
    FILEPATH_PREDICT['filename'] = None
    FILEPATH_PREDICT['original'] = None
    FILEPATH_PREDICT['object_detection'] = None
    FILEPATH_PREDICT['crop_object'] = None
    FILEPATH_PREDICT['text_detection'] = None

def set_path_predict_original(path, filename):
    global FILEPATH_PREDICT
    FILEPATH_PREDICT['filename'] = filename
    FILEPATH_PREDICT['original'] = path
           
def get_path_predict_original():
    global FILEPATH_PREDICT
    return FILEPATH_PREDICT['original'], FILEPATH_PREDICT['filename']

def set_path_predict_result(pathOB, pathCrop, pathTD):
    global FILEPATH_PREDICT
    FILEPATH_PREDICT['object_detection'] = pathOB
    FILEPATH_PREDICT['crop_object'] = pathCrop
    FILEPATH_PREDICT['text_detection'] = pathTD
    
def get_path_predict_result():
    global FILEPATH_PREDICT
    return FILEPATH_PREDICT['object_detection'], FILEPATH_PREDICT['crop_object'], FILEPATH_PREDICT['text_detection']

def set_path_train(pathRes, pathVal, pathPlot, pathModel):
    global FILEPATH_TRAIN
    FILEPATH_TRAIN['result'] = pathRes
    FILEPATH_TRAIN['val'] = pathVal
    FILEPATH_TRAIN['plot'] = pathPlot
    FILEPATH_TRAIN['model'] = pathModel
    
def set_path_train_status(status):
    FILEPATH_TRAIN['status'] = status
    
def get_path_train_status():
    return FILEPATH_TRAIN['status']

def set_res_train(model_name, model, accuracy, precision, recall):
    global RESULT_TRAIN
    RESULT_TRAIN['model_name'] = model_name + '.pt'
    RESULT_TRAIN['model'] = model
    RESULT_TRAIN['accuracy'] = accuracy
    RESULT_TRAIN['precision'] = precision
    RESULT_TRAIN['recall'] = recall

def get_path_train_plot():
    global FILEPATH_TRAIN
    return FILEPATH_TRAIN['plot']

def get_train_model():
    global RESULT_TRAIN
    global FILEPATH_TRAIN
    return RESULT_TRAIN['model_name'], FILEPATH_TRAIN['model']

def get_res_train_acc():
    global RESULT_TRAIN
    return RESULT_TRAIN['accuracy']

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/read-plate")
def read_plate():
    # dropzone.config['MAX_FILES'] = 1
    # dropzone.config['DROPZONE_MAX_FILES'] = 1
    filepath, filename = get_path_predict_original()
    if filepath != None:
        print('\n Mengakses read plate dengan filepath...\n')
        print(f'\n filepath : {filepath}...\n')
        return render_template("read_plate.html", filepath=filepath, filename=filename)
    # elif filepath == 'error' and filename == 'error':
    #     print('\n Silahkan upload file jpg...\n')
    #     flash('Please upload images in JPG or PNG format!', 'error')
    #     set_path_none()
    #     return redirect("/read-plate")
    else:
        print('\n Mengakses read plate langsung...\n')
        return render_template("read_plate.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        filename = secure_filename(f.filename)
        if f and allowed_file(filename):
            print('\n Mengecek ekstensi file...\n')
            ext = get_file_extension(filename)
            if ext != "txt":
                print('\n Proses dimulai...\n')
                filepath = os.path.join(app.config['UPLOADED_FOLDER_PREDICT'], filename)
                set_path_predict_original(filepath, filename)
                print('\n Path berhasil diubah...\n')
                f.save(filepath)
                print('\n File berhasil disimpan...\n')
                print('\n--------------')
                # print(UPLOAD_FOLDER)
                print(filepath)
                print(filename)
                print('\n\n')
                # flash('File successfully uploaded')
                return redirect("/read-plate")
            else:
                print('\n FIle bukan PNG / JPG...\n')
                set_path_predict_original('error', 'error')
                # flash('Please upload images in JPG or PNG format!', 'error')
                return redirect("/read-plate")
        else:
            print('\n FIle sembarangan...\n')
            set_path_predict_original('error', 'error')
            # flash('Please upload images in JPG or PNG format!', 'error')
            return redirect("/read-plate")
    else:
        return redirect("/read-plate")
    
@app.route("/result", methods=['GET', 'POST'])
def result():
    print('\n--------------')
    print('\n\n Mengakses /result....\n')
    filepath, filename = get_path_predict_original()
    if filepath != None and filepath != 'error' and filename != 'error' and request.form['start-process-read']:
        print('\n\nStart Process Data....\n')
        start_process_read = request.form['start-process-read']
        print(f'\n filepath : {filepath}...\n')
        print(f'\n Start Predict Image...\n')
        result_text_plate = predict(filepath, filename)
        pathOB, pathCrop, pathTD = get_path_predict_result()
        return render_template('result.html', filepath=filepath, filename=filename, pathOB=pathOB, pathCrop=pathCrop, pathTD=pathTD, result_text_plate=result_text_plate, start_process_read=start_process_read)
    elif filepath == 'error' and filename == 'error' and request.form['start-process-read']:
        print('\n Silahkan upload file jpg...\n')
        set_path_none()
        flash('Please upload images in JPG or PNG format!', 'error')
        return redirect("/read-plate")
    else:
        print('\n\n Mengembalikkan ke halaman read_plate....\n')
        return redirect('/read-plate')
    
@app.route("/download-result")
def download_result():
    print('\n--------------')
    print('\n\n Mengakses /download-result....\n')
    filepath, filename = get_path_predict_original()
    if filepath != None:
        pathOB, pathCrop, pathTD = get_path_predict_result()
        filename = 'Result_' + filename
        return send_file(
            pathTD,
            download_name=filename,
            as_attachment=True,
        )   
    else:
        print('\n\n Mengembalikkan ke halaman read_plate....\n')
        return redirect('/read-plate')
    
@app.route("/read-again")
def read_again():
    set_path_none()
    return redirect('/read-plate')

@app.route("/upgrade-model")
def upgrade_model():
    # dropzone.config['MAX_FILES'] = 1
    # dropzone.config['DROPZONE_MAX_FILES'] = 1
    filepath, filename = get_path_predict_original()
    if filepath != None:
        print('\n Mengakses upgrade_model dengan filepath...\n')
        print(f'\n filepath : {filepath}...\n')
        return render_template("upgrade_model.html", filepath=filepath, filename=filename)
    else:
        print('\n Mengakses upgrade_model langsung...\n')
        return render_template("upgrade_model.html")
    
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
            set_path_train_status('start')
            print('\n--------------')
            # print(UPLOAD_FOLDER)
            print(filepath)
            print(filename)
            print('\n\n')
            # flash('File successfully uploaded')
            return redirect("/upgrade-model")
        else: 
            print('\n File bukan sembarangan...\n')
            set_path_train_status('cancel')
            return redirect("/upgrade-model")
    else:
        return redirect("/upgrade-model")

@app.route("/train-model", methods=['GET', 'POST'])
def train_model():
    print('\n--------------')
    print('\n\n Mengakses /train....\n')
    if request.form['start-train'] == 'True':
        epoch = int(request.form['epoch'])
        confidence = int(request.form['confidence'])
        print('\n\n Start Train Data....\n')
        model, accuracy, precision, recall = train(epoch, confidence)
        plotTrain = get_path_train_plot()
        return render_template('score.html', model=model, accuracy=accuracy, precision=precision, recall=recall, plotTrain=plotTrain)
    elif request.form['start-train'] == 'False':
        print('\n\n Mengembalikkan ke halaman upgrade model....\n')
        return redirect('/upgrade-model')
    else:
        print('\n\n Mengembalikkan ke halaman upgrade model....\n')
        return redirect('/upgrade-model')

    
@app.route("/save-model")
def save_model():
    print('\n--------------')
    print('\n\n Mengakses /save-model....\n')
    modelName, pathModel = get_train_model()
    if pathModel != None:
        modelName = 'Model_' + modelName
        print(f'\n\n pathModel: {pathModel}\n----')
        return send_file(
            pathModel,
            download_name=modelName,
            as_attachment=True,
        )   
    else:
        print('\n\n Mengembalikkan ke halaman upgrade model....\n')
        return redirect('/score')

# @app.route("/score")
# def score():
#     accuracy = get_res_train_acc()
#     plotTrain = get_path_train_plot()
#     return render_template('score.html', accuracy=accuracy, plotTrain=plotTrain)

@app.route("/score")
def score():
    return render_template('score.html')

@app.route("/res")
def res():
    return render_template('result.html')

# @app.route("/result", methods=['GET', 'POST'])
# def result():
#     if request.method == 'POST':
#         if request.files['input-image']:
#             file = request.files['input-image']
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join("static", "upload", filename)
#                 file.save(filepath)
#                 print('\n--------------')
#                 # print(UPLOAD_FOLDER)
#                 print(filepath)
#                 print(filename)
#                 print('\n\n')
#                 # flash('File successfully uploaded')
#                 return render_template("result.html", filename = filename, filepath=filepath)
#     else:
#         return redirect("/read-plate")

if __name__ == '__main__':
    app.run(debug=True)