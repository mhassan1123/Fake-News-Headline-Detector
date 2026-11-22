Fake News Headline Detector

A project that analyzes news headlines using Machine Learning(ML) and classification algorithms. It compares Naive Bayes and Logistic Regression, uses TF-IDF for text representation, and applies K-means clustering to explore headline topics.

The project also provides an interactive Streamlit application for headline analysis, model explanations, and topic exploration.

Project Features

Classifies headlines as Fake, Real, or Insufficient Evidence.

Uses TF-IDF to convert text into numerical features.

Compares Naive Bayes and Logistic Regression classifiers.

Uses a selected Logistic Regression model for final predictions.

Provides confidence scores and influential-word explanations.

Detects possible spelling issues and provides advisory suggestions without automatically changing user input.

Uses K-means clustering to explore news topics.

Includes an interactive Streamlit web application.

Demonstrates object-oriented programming through inheritance, encapsulation, and method overriding.

Technologies Used

Python

Pandas and NumPy

Scikit-learn

SciPy

Joblib

Matplotlib

Streamlit

TF-IDF

Naive Bayes

Logistic Regression

K-means Clustering

Dataset

The project uses the cleaned dataset:

data/processed/final_headlines_cleaned_v2.csv

The dataset contains 82,823 headlines.

Dataset Splits

Split

Number of Headlines

Training

53,006

Validation

13,252

Testing

16,565

Total

82,823

The training split is used for model training, validation is used for model comparison and selection, and the held-out test split is used for final evaluation.

Machine Learning Approach

1. Text Representation

TF-IDF (Term Frequency–Inverse Document Frequency) converts news headlines into numerical feature vectors.

The final classification pipeline uses 10,000 TF-IDF features.

2. Classification Models

Two classification algorithms were compared:

Multinomial Naive Bayes

Logistic Regression

Logistic Regression was selected as the final classification model.

The final model uses a classification threshold of 0.477.

3. Model Evaluation

The selected model was evaluated on the held-out test dataset.

Metric

Test Result

Macro F1-score

0.8144

ROC-AUC

0.8972

The test dataset was kept separate from model selection and threshold tuning.

4. Topic Clustering

K-means clustering is used to group headlines according to their text features.

The application uses previously generated clustering results to display topic summaries and explore news headline groups.

Object-Oriented Design

The project includes four main classes:

HeadlineEnvironment: handles headline input and validation.

TextClassifier: parent class containing shared classification functionality.

NaiveBayesAgent: inherits from TextClassifier and implements Naive Bayes prediction.

LogisticRegressionAgent: inherits from TextClassifier and implements Logistic Regression prediction.

The implementation demonstrates:

Encapsulation

Inheritance

Method overriding

Code reuse

Streamlit Application

The project provides three main application sections:

Analyze Headline: Enter a headline and view its prediction, confidence, influential words, and warnings.

Explore News Topics: Explore existing K-means clusters and their summaries.

About Project: View information about the project, model, and its limitations.

Run the Application

Start the application from the project root directory:

python -m streamlit run app.py

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

Installation

1. Clone the Repository

git clone https://github.com/mhassan1123/Fake-News-Headline-Detector
cd Fake-News-Headline-Detector

https://github.com/mhassan1123/Fake-News-Headline-Detector

2. Create a Virtual Environment

python -m venv venv

Activate it on Windows PowerShell:

.\venv\Scripts\Activate.ps1

3. Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run the Application

python -m streamlit run app.py

Project Structure

Fake-News-Headline-Detector/
│
├── app.py
├── requirements.txt
├── README.md
├── generate_charts.py
│
├── data/
│   ├── processed/
│   │   └── final_headlines_cleaned_v2.csv
│   └── splits_cleaned_v2/
│
├── experiments/
│   ├── run_experiments.py
│   ├── train_naive_bayes.py
│   ├── train_logistic_regression.py
│   ├── finalize_model_selection.py
│   └── evaluate_final_model.py
│
├── outputs/
│   └── cleaned_v2/
│       ├── models/
│       ├── clustering/
│       ├── reproducible_experiment/
│       └── audits/
│
├── src/
│   ├── agents.py
│   ├── inference.py
│   ├── explain.py
│   ├── clustering.py
│   ├── data_loader.py
│   └── preprocessing.py
│
└── docs/
    ├── agent_design.md
    ├── learning_agent_architecture.md
    ├── class_diagram.md
    ├── dataset_schema.md
    └── charts/

The application depends on the saved model and vectorizer artifacts in outputs/cleaned_v2/. Keep the required artifacts and their directory structure intact.

Documentation and Visualizations

Additional project documentation is available in the docs/ directory:

Project Report – Word Document

PEAS and Task Environment Classification

Learning Agent Architecture

UML Class Diagram and OOP Concepts

Dataset Schema

Generated evaluation charts are available in docs/charts/:

model_comparison_validation.png

final_test_confusion_matrix.png

cluster_distribution.png

Limitations

The system analyzes headline text, not the complete news article.

Predictions are based on patterns learned from the training dataset.

Model confidence is not the same as factual certainty.

Misspellings, incomplete headlines, unfamiliar names, and out-of-vocabulary words may reduce prediction reliability.

Spelling suggestions are advisory and do not automatically modify the original headline.

The application does not independently verify news sources or establish whether a claim is factually true.

Important: This project is an educational machine learning system, not a professional fact-checking service. Its predictions should not be treated as proof that a news story is true or false.

License

Add the appropriate license information if this project is intended for public distribution.