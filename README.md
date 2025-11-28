📘 Student Performance Prediction
--

A machine learning web application that predicts whether a student will pass or fail, based on demographic and academic features.
The project includes full model training, evaluation, and deployment using Flask and Render.



🚀 Live Web App
--

🔗 https://studentperformanceproject-2.onrender.com/

--
📊 Model Comparison
--

Multiple machine learning models were trained and evaluated using accuracy, precision, recall, and F1-score. Below is the comparison table summarizing their performance:

| Model                   | Accuracy | Precision | Recall | F1-Score |
| ----------------------- | -------- | --------- | ------ | -------- |
| **Logistic Regression** | **0.80** | 0.79      | 0.80   | 0.79     |
| Random Forest           | 0.78     | 0.76      | 0.77   | 0.76     |
| Decision Tree           | 0.72     | 0.70      | 0.71   | 0.70     |
| K-Nearest Neighbors     | 0.74     | 0.73      | 0.74   | 0.73     |
| AdaBoost                | 0.76     | 0.75      | 0.75   | 0.75     |
| XGBoost                 | 0.77     | 0.76      | 0.77   | 0.76     |


--
🧠 Model Used
--
The deployed model is:

⭐ Logistic Regression (best-performing model)

1)Selected after comparing multiple models including Random Forest, Decision Tree, KNN, XGBoost, and AdaBoost.

2)Logistic Regression achieved the highest balanced performance and was saved as:
  student_model.pkl
  
--
📊 Dataset Description
--

This project uses the Students Performance in Exams dataset from Kaggle, which contains demographic, socioeconomic, and academic information about students.
It includes 1,000 records, where each row represents one student.

🔗 Dataset Source:
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
--
Features
--

1)gender – Male or Female

2)race/ethnicity – Student’s ethnic group (A–E)

3)parental level of education – Highest education level of the student's parents

4)lunch – Type of lunch support (standard or free/reduced)

5)test preparation course – Completed or not completed

6)math score, reading score, writing score – Exam performance (0–100)

Target Variable

-Pass/Fail Classification

   -Created using the mean of math, reading, and writing scores

   -Students with an average ≥ 60% → Pass

  -Students with an average < 60% → Fail

This dataset is widely used to explore how demographic and socioeconomic factors influence academic outcomes and is ideal for building classification models.

--
🔧 Tech Stack
--
1)Python

2)Flask

3)Scikit-learn

4)Pandas

5)HTML / CSS

6)Render (Deployment)

--
🛠️ How the Model Was Built
--
1)Data preprocessing using OneHotEncoding and StandardScaler

2)Building a full ML pipeline

3)Training multiple ML models

4)Selecting the best performing model (Logistic Regression)

5)Saving the model pipeline as student_model.pkl

6)Deploying using Flask + Gunicorn on Render

--
## 📂 Project Structure

```plaintext
├── app.py
├── student_model.pkl
├── requirements.txt
├── templates/
│   └── index.html
├── StudentsPerformance.csv
└── Model_Training.ipynb
```


📝 Usage
--
Enter student attributes in the web form
→ The model predicts Pass or Fail in real-time.
