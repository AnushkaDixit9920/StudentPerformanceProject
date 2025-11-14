from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)


model_path = os.path.join("artifacts", "student_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = {
            "gender": request.form["gender"],
            "race/ethnicity": request.form["race/ethnicity"],
            "parental level of education": request.form["parental level of education"],
            "lunch": request.form["lunch"],
            "test preparation course": request.form["test preparation course"]
        }

        
        input_df = pd.DataFrame([data])

  
        prediction = model.predict(input_df)[0]
        result = "Pass ✔" if prediction == 1 else "Fail ✘"

        return render_template("index.html", prediction_text=f"Predicted Result: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
