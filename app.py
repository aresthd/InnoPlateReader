from flask import Flask, render_template

app = Flask(__name__)

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
	return render_template("read_plate.html")

@app.route("/result")
def result():
	return render_template("result.html")

if __name__ == '__main__':
	app.run(debug=True)