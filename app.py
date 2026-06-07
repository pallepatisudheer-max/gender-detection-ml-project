from flask import Flask, request
import pickle

app = Flask(__name__)

with open("gender_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return """
    <h2>Gender Prediction</h2>
    <form action="/predict" method="post">
        <input type="text" name="name" placeholder="Enter Name">
        <button type="submit">Predict</button>
    </form>
    """

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    prediction = model.predict([name])[0]
    return f"Predicted Gender: {prediction}"

if __name__ == "__main__":
    app.run(debug=True)