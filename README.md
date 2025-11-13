# SMS Spam Classification Project

## 1. Introduction

This project aims to build a system that classifies SMS messages into two categories: **Spam** (unwanted messages) and **Ham** (legitimate messages), using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The goal is to create a complete pipeline from raw data preprocessing to deploying the trained model on a web application.

The project is conducted as part of the course **Programming for Data Science** at the **National Economics University (NEU)**.

---

## 2. Team Members and Responsibilities

| Member | Student ID | Main Responsibilities | Assigned Files |
|---------|-------------|------------------------|----------------|
| **Dương Hữu Tuấn Anh** | 11245832 | Prepared the project report, wrote the README.md, described the pipeline, project structure, and provided setup instructions. | README.md, report.pdf |
| **Vũ Anh Sơn** |  | Backend – Developed the Streamlit application, loaded `spam_model.pkl` and `vectorizer.pkl`, implemented prediction logic, and integrated with the frontend. | app.py |
| **Tạ Ngọc Ánh** |  | Frontend – Designed the Streamlit web interface, visualized prediction results, and created WordClouds for Spam/Ham keywords. | app.py, assets/ |
| **Nguyễn Thị Dương** |  | Built the `ManualVectorizer` (Bag-of-Words), saved processed datasets, and coordinated Step 2 of the pipeline. | vectorizer.py |
| **Trần Nguyên Khôi** |  | Implemented the tokenization function, handled raw CSV data reading and preprocessing steps. | preprocess.py |
| **Đỗ Quốc Trung** |  | Performed system testing, entered test data, and reported UI or logic-related bugs. | test_examples.txt |

---

## 3. Problem Description

- **Input:** A text string (SMS message)

- **Output:**
  - Predicted label — `Spam` or `Ham`
  - Confidence score (e.g., `Spam – 99.2% confidence`)
  - If predicted as `Spam`, the system highlights keywords that likely triggered the spam classification (e.g., `"win"`, `"free"`, `"click"`)

- **Goal:** Build a machine learning model capable of automatically distinguishing spam messages from legitimate ones based on text content.
---

## 4. Processing Pipeline

The project is divided into four main stages:

### 4.1 Data Preprocessing
- Read the original dataset `spam.csv`.
- Normalize text: convert to lowercase, retain only alphabetic characters and spaces.
- Remove common English stopwords.
- Tokenize text into words.
- Split data into training (80%) and testing (20%) sets using **stratified sampling**.

### 4.2 Vocabulary Building and Text Vectorization
- Build a vocabulary of the 3000 most frequent words from the training data.
- Convert each message into a count vector using the **Bag-of-Words** approach.
- Save the vectorizer with `joblib` as `artifacts/vectorizer.pkl`.

### 4.3 Model Training
- Models used: **Support Vector Machine (SVM)** and optionally *Multinomial Naive Bayes or Logistic Regression*.
- Train the model on the training set and evaluate it on the testing set.
- Evaluation metrics include: Accuracy, Precision, Recall, and F1-score.
- Save the trained model as artifacts/spam_model.pkl.

### 4.4 Web Application Deployment
- The web application is developed using **Streamlit**.
- Users can input any SMS message for classification.
- The system preprocesses, vectorizes, and predicts the message label in real time.
- The prediction result (Spam/Ham) is displayed in a clear and user-friendly interface.

---

## 5. Model Evaluation Results

*(To be updated after training and testing the model.)*

---
## 6. Project Directory Structure

```bash
project/
│
├── data/
│   ├── raw/                # Original dataset (spam.csv)
│   └── processed/          # Preprocessed train/test sets
│
├── artifacts/
│   ├── vectorizer.pkl      # Saved vectorizer
│   └── spam_model.pkl      # Trained ML model
│
├── templates/              # HTML templates for frontend
├── static/                 # CSS and JS files
├── app.py                  # Python backend application
├── train.py                # Model training script
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation
````

## 7. Installation and Execution

### 7.1 System Requirements

* Python 3.8 or higher
* pip or conda for dependency installation

### 7.2 Install Dependencies

Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### 7.3 Train Model (Optional)

If you wish to retrain the model from scratch using the raw dataset:

```bash
python train.py
```

### 7.5 Libraries Used

* **pandas** – Data manipulation and CSV processing
* **numpy** – Numerical computation
* **scikit-learn** – Machine learning algorithms and metrics
* **joblib** – Saving and loading models/vectorizers
* **wordcloud**, **matplotlib** – Keyword visualization and plots


