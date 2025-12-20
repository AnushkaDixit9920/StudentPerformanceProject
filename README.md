🧠 Student Performance Prediction Using Machine Learning
--
📄 Overview
--
This project focuses on predicting whether a student will pass or fail based on demographic, socioeconomic, and academic attributes such as gender, parental education, lunch type, test preparation, and exam scores.
The objective is to understand how non-academic factors influence student outcomes and to build a robust machine learning model that can predict performance accurately. The project also includes deployment as a live web application, enabling real-time predictions.

--
🎯 Objectives
--
1)To analyze how demographic and socioeconomic factors affect student performance

2)To identify patterns that contribute to academic success or failure

3)To compare multiple machine learning algorithms for classification

4)To deploy a real-time prediction system using a web interface

5)To demonstrate an end-to-end ML workflow: data → model → deployment

--
🧩 Dataset
--
Source:

🔗 https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

Description:

The dataset contains 1,000 student records, where each record represents an individual student’s background and academic performance.

Features:

1)Gender – Male / Female

2)Race/Ethnicity – Group A to E

3)Parental Level of Education – Highest education of parents

4)Lunch – Standard or Free/Reduced

5)Test Preparation Course – Completed / Not Completed

6)Math Score (0–100)

7)Reading Score (0–100)

8)Writing Score (0–100)

--
🎯 Target Variable
--
Pass / Fail Classification

1)Average of math, reading, and writing scores calculated

2)Average ≥ 60% → Pass

3)Average < 60% → Fail

--
⚙️ Methodology
--
🧹 Data Cleaning & Preprocessing

1)Checked and handled missing values

2)Converted categorical variables using One-Hot Encoding

3)Scaled numerical features using StandardScaler

4)Created a new binary target variable (Pass / Fail)

--
📊 Exploratory Data Analysis (EDA)
--
1)Analyzed score distributions across demographic groups

2)Studied the effect of parental education and lunch type

3)Observed performance gaps based on socioeconomic factors

4)Identified trends linking test preparation to higher scores
--
📊 Exploratory Data Analysis (EDA)
--
1)Analyzed score distributions across demographic groups

2)Studied the effect of parental education and lunch type

3)Observed performance gaps based on socioeconomic factors

4)Identified trends linking test preparation to higher scores

--
⚒️ Feature Engineering
--
1)Encoded categorical features

2)Normalized numerical scores

3)Built a complete machine learning pipeline

--
🤖 Model Training
--
Multiple machine learning models were trained and compared:

1)Logistic Regression

2)Random Forest

3)Decision Tree

4)K-Nearest Neighbors (KNN)

5)AdaBoost

6)XGBoost

--
📈 Model Evaluation & Comparison
--
| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.80     | 0.79      | 0.80   | 0.79     |
| Random Forest       | 0.78     | 0.76      | 0.77   | 0.76     |
| Decision Tree       | 0.72     | 0.70      | 0.71   | 0.70     |
| KNN                 | 0.74     | 0.73      | 0.74   | 0.73     |
| AdaBoost            | 0.76     | 0.75      | 0.75   | 0.75     |
| XGBoost             | 0.77     | 0.76      | 0.77   | 0.76     |

--
🧠 Final Model Selected
--
⭐ Logistic Regression

1)Achieved the best balanced performance

2)High interpretability and stability

3)Saved as: student_model.pkl

4)Used in the deployed web application

--
🌐 Live Web Application
--
🔗 Live App:

https://studentperformanceproject-2.onrender.com/

📝 Usage

1)Enter student details in the web form

2)Submit the data

3)Get real-time prediction: Pass or Fail

--
💡 Key Insights
--
1)Students who completed test preparation courses perform better

2)Parental education level has a strong influence on outcomes

3)Students with free/reduced lunch show slightly lower performance trends

4)Academic scores remain the strongest predictors, but socioeconomic factors add meaningful context

--
🧰 Tech Stack
--
1)Language: Python

2)Libraries: Pandas, NumPy, Scikit-learn

3)Web Framework: Flask

4)Frontend: HTML, CSS

5)Deployment: Render

6)Version Control: Git & GitHub

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
🚀 Future Scope
--

1)Extend prediction to grade-level classification

2)Add feature importance visualizations in the web app

3)Build a dashboard for educators to monitor trends

4)Experiment with deep learning models

--
👩‍💻 Author
--

Anushka Dixit
Computer Science & AI Engineering Student

