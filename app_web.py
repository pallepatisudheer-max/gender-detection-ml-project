from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("gender_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        name = request.form["name"]
        prediction = model.predict([name])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    