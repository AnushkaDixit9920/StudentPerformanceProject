📘 Student Performance Prediction

A machine learning web application that predicts whether a student will pass or fail, based on demographic and academic features.
The project includes full model training, evaluation, and deployment using Flask and Render.
--

🚀 Live Web App

🔗 https://studentperformanceproject-2.onrender.com/
--
🧠 Model Used

The deployed model is:

⭐ Logistic Regression (best-performing model)

1)Selected after comparing multiple models including Random Forest, Decision Tree, KNN, XGBoost, and AdaBoost.

2)Logistic Regression achieved the highest balanced performance and was saved as:
  student_model.pkl
--
📊 Features Considered

1)Gender

2)Race/ethnicity

3)Parental level of education

4)Lunch type

5)Test preparation course
--
🔧 Tech Stack

1)Python

2)Flask

3)Scikit-learn

4)Pandas

5)HTML / CSS

6)Render (Deployment)
--
🛠️ How the Model Was Built

1)Data preprocessing using OneHotEncoding and StandardScaler

2)Building a full ML pipeline

3)Training multiple ML models

4)Selecting the best performing model (Logistic Regression)

5)Saving the model pipeline as student_model.pkl

6)Deploying using Flask + Gunicorn on Render
--
📂 Project Structure
--
├── app.py
├── student_model.pkl
├── requirements.txt
├── templates/
│   └── index.html
├── StudentsPerformance.csv
└── Model_Training.ipynb

--

📝 Usage

Enter student attributes in the web form
→ The model predicts Pass or Fail in real-time.
