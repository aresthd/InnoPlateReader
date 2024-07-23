from werkzeug.utils import secure_filename
import os
import cv2
import imutils
import easyocr
from ultralytics import YOLO
from datetime import datetime


# Train
def train(model, config, config_train, epoch, confidence):
    basedirTraining = os.path.join('static', 'training')
    confidence = confidence / 100

    date = f'{str(datetime.now().day)}_{str(datetime.now().month)}_{str(datetime.now().year)}'
    time = f'{str(datetime.now().hour)}_{str(datetime.now().minute)}_{str(datetime.now().second)}'
    name = f'{date}_{time}'
    
    try:
        result = model.train(data=config, epochs=epoch, project=basedirTraining, name=name)
        print(result)
        metrics = model.val(conf=confidence)
        # pathPlot = "static/dist/img"
        # metrics.confusion_matrix.plot(normalize=False, save_dir=pathPlot)
        
        pathRun = result.save_dir
        pathVal = metrics.save_dir
        pathPlot = os.path.join(pathRun, "confusion_matrix.png")
        pathModel = os.path.join(pathRun, 'weights', 'best.pt')
        
        cf = metrics.confusion_matrix.matrix
        tp = cf[0][0]
        fp = cf[0][1]
        fn = cf[1][0]
        tn = cf[1][1]
        accuracy = ((tp + tn) / (tp+fp+fn+tn)) * 100
        precision = (tp / (tp+fp)) * 100
        recall = (tp /(tp+fn)) * 100
        
        set_all_conf_train(config_train, 'success', name, model, pathModel, pathRun, pathVal, pathPlot, accuracy, precision, recall)
        return model, accuracy, precision, recall
    except IndexError as ae:
        accuracy = 0
        precision = 0
        recall = 0
        set_conf_train(config_train, 'status', 'add data')
        return model, accuracy, precision, recall
    except FileNotFoundError as ae:
        accuracy = 0
        precision = 0
        recall = 0
        set_conf_train(config_train, 'status', 'no image')
        return model, accuracy, precision, recall
    except AttributeError as ae:
        accuracy = 0
        precision = 0
        recall = 0
        set_conf_train(config_train, 'status', 'no label')
        return model, accuracy, precision, recall
    
# Predict
def predict(model, config_predict, path_image, filename):
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
    
    set_all_conf_predict(config_predict, filename, path_image, pathImgOb, pathImgCrop, pathImgTD)
    return textToWrite

def allowed_file(filename, allowed_ext):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_ext

def get_file_extension(file_name):
    try:
        file_extension = file_name.split('.')[-1]
        return file_extension.lower() 
    except IndexError:
        return None

def set_all_conf_predict(config_predict, filename=None, path_ori=None, path_ob=None, path_crop=None, path_td=None):
    config_predict['filename'] = filename
    config_predict['path_ori'] = path_ori
    config_predict['path_ob'] = path_ob
    config_predict['path_crop'] = path_crop
    config_predict['path_td'] = path_td

def set_all_conf_train(config_train, status=None, model_name=None, model=None, path_model=None, path_run=None, path_val=None, path_plot=None, accuracy=None,  precision=None,  recall=None):
    config_train['status'] = status
    config_train['model_name'] = model_name
    config_train['model'] = model
    config_train['path_model'] = path_model
    config_train['path_run'] = path_run
    config_train['path_val'] = path_val
    config_train['path_plot'] = path_plot
    config_train['accuracy' ] = accuracy
    config_train['precision' ] = precision
    config_train['recall'] = recall

def set_conf_predict(config_predict, keyword, value):
    config_predict[keyword] = value

def set_conf_train(config_train, keyword, value):
    config_train[keyword] = value

def get_conf_predict(config_predict, keyword):
    return config_predict[keyword]

def get_conf_train(config_train, keyword):
    return config_train[keyword]
