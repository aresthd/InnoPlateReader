from flask import Flask, render_template, request, redirect, url_for
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
import os

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.secret_key = "secret key"
app.config.update(
    UPLOADED_FOLDER = os.path.join('static', 'uploads'),
    DROPZONE_TIMEOUT = 5 * 60 * 1000,
    DROPZONE_ADD_REMOVE_LINKS = True
)
dropzone = Dropzone(app)

# UPLOAD_FOLDER = "\static\upload"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FILEPATH_READ_PLATE = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def set_filepath_read(path):
    global FILEPATH_READ_PLATE
    FILEPATH_READ_PLATE = path
           
def get_filepath_read():
    global FILEPATH_READ_PLATE
    return FILEPATH_READ_PLATE

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

@app.route("/read_plate")
def read_plate():
    # dropzone.config['MAX_FILES'] = 1
    # dropzone.config['DROPZONE_MAX_FILES'] = 1
    filepath = get_filepath_read()
    if filepath != None:
        print('\n Mengakses read plate dengan filepath...\n')
        print(f'\n filepath : {filepath}...\n')
        return render_template("read_plate.html", filepath=filepath)
    else:
        print('\n Mengakses read plate langsung...\n')
        return render_template("read_plate.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            print('\n Proses dimulai...\n')
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOADED_FOLDER'], filename)
            set_filepath_read(filepath)
            print('\n Path berhasil diubah...\n')
            f.save(filepath)
            print('\n File berhasil disimpan...\n')
            print('\n--------------')
            # print(UPLOAD_FOLDER)
            print(filepath)
            print(filename)
            print('\n\n')
            # flash('File successfully uploaded')
            return redirect("/read_plate")
    else:
        return redirect("/read_plate")
    
@app.route("/result", methods=['GET', 'POST'])
def result():
    print('\n--------------')
    print('\n\n Mengakses /result....\n')
    filepath = get_filepath_read()
    if filepath != None and request.form['start-process-read']:
        print('\n\nStart Process Data....\n')
        start_process_read = request.form['start-process-read']
        print(f'\n filepath : {filepath}...\n')
        return render_template('result.html', filepath=filepath, start_process_read=start_process_read)
    else:
        print('\n\n Mengembalikkan ke halaman read_plate....\n')
        return redirect('/read_plate')

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
#         return redirect("/read_plate")

if __name__ == '__main__':
    app.run(debug=True)