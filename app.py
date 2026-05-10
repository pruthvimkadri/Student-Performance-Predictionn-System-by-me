from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the NEW model file
model = pickle.load(open("student_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    input_data = {}

    if request.method == "POST":
        try:
            input_data = {
                "age": float(request.form["age"]),
                "gender": request.form["gender"],
                "school_type": request.form["school_type"],
                "parent_education": request.form["parent_education"],
                "study_hours": float(request.form["study_hours"]),
                "attendance_percentage": float(request.form["attendance_percentage"]),
                "internet_access": request.form["internet_access"],
                "travel_time": request.form["travel_time"],
                "extra_activities": request.form["extra_activities"],
                "study_method": request.form["study_method"],
                "math_score": float(request.form["math_score"]),
                "science_score": float(request.form["science_score"]),
                "english_score": float(request.form["english_score"])
            }

            input_df = pd.DataFrame([input_data])

            pred = model.predict(input_df)[0]
            prediction = str(pred)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                confidence = round(max(proba) * 100, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        input_data=input_data
    )

if __name__ == "__main__":
    app.run(debug=True)