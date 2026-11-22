# Intelligent Agent Design

## 1. Introduction

The Fake News Headline Detector is a machine learning-based intelligent system that analyzes user-provided news headlines and predicts whether they are likely to be Fake or Real.

The system uses TF-IDF for text feature extraction and a trained Logistic Regression model for its final classification. It also uses input validation, spelling suggestions, and influential-word explanations to make its predictions more understandable.

If the input is empty, invalid, or contains no recognized vocabulary, the system returns Insufficient Evidence instead of forcing a classification.

This section describes the intelligent agent using the PEAS framework and classifies its task environment.

## 2. PEAS Description

PEAS stands for Performance Measure, Environment, Actuators, and Sensors. It is used to describe the task performed by an intelligent agent.

### 2.1 Performance Measure (P)

Performance measures evaluate how successfully the intelligent agent performs its task.

The performance of the Fake News Headline Detector is evaluated using:

- Accuracy: Measures the proportion of correctly classified headlines.
- Precision: Measures how many headlines predicted as a particular class actually belong to that class.
- Recall: Measures how many actual examples of a class are correctly identified.
- Macro F1-score: Evaluates the balance between precision and recall across both classes.
- ROC-AUC: Measures the model's ability to distinguish between the Fake and Real classes.
- Input handling: Checks whether the system responds appropriately to empty, invalid, misspelled, and unfamiliar inputs.
- Explanation quality: Provides influential words to help users understand the model's prediction.

The selected Logistic Regression model achieved a test Macro F1-score of 0.8144 and a test ROC-AUC of 0.8972 at the selected classification threshold of 0.477.

The system also aims to provide understandable outputs and avoid presenting uncertain predictions as confirmed facts.

### 2.2 Environment (E)

The environment consists of the external and internal resources with which the intelligent agent interacts.

The environment of this project includes:

- Users interacting with the Streamlit application.
- News headlines entered by users.
- The trained Logistic Regression and Naive Bayes models.
- The TF-IDF vectorizer and its learned vocabulary.
- Saved model and experiment files.
- Existing news topic-clustering results.
- The computer system on which the application runs.

The agent processes the headline supplied by the user and returns a classification result.

### 2.3 Actuators (A)

Actuators are the mechanisms through which an intelligent agent communicates or performs its actions.

The actuators of this project include:

- Displaying a Fake or Real prediction.
- Displaying Insufficient Evidence when a reliable prediction cannot be produced.
- Showing the model confidence score.
- Displaying influential words behind the prediction.
- Showing input validation warnings.
- Providing possible spelling suggestions.
- Displaying news topic clusters, their important words, and example headlines.

These outputs are presented through the Streamlit user interface.

### 2.4 Sensors (S)

Sensors allow an intelligent agent to receive information from its environment.

The sensors and input-processing mechanisms in this project include:

- The Streamlit text input field, which receives a headline from the user.
- Input validation, which checks whether the submitted headline is valid.
- The text preprocessing and TF-IDF vectorization pipeline, which converts the headline into numerical features.
- Access to saved clustering results when the user explores news topics.

The resulting TF-IDF features are supplied to the trained classification model.

### 2.5 PEAS Summary

| PEAS Component | Description |
|---|---|
| Performance Measure | Accuracy, precision, recall, Macro F1-score, ROC-AUC, input handling, and explanation quality |
| Environment | Users, submitted headlines, Streamlit application, trained models, vectorizer, and saved clustering results |
| Actuators | Predictions, confidence scores, influential words, warnings, spelling suggestions, and cluster summaries |
| Sensors | User text input, input validation, and text feature extraction |

## 3. Task Environment Classification

A task environment describes the conditions under which an intelligent agent operates.

The Fake News Headline Detector can be classified using the following properties.

### 3.1 Partially Observable

Classification: Partially Observable

The agent receives the headline entered by the user but does not have access to all the information required to establish whether the underlying news story is actually true.

For example, the system may receive a headline without the full article, original source, supporting evidence, or independent verification.

Therefore, the actual truth of the news is not fully observable to the agent.

### 3.2 Deterministic

Classification: Deterministic at inference time

The trained Logistic Regression model produces a repeatable output for the same input when the model, vectorizer, and configuration remain unchanged.

Therefore, the inference process is deterministic.

However, a deterministic prediction does not guarantee that the predicted label is factually correct. The model can still make classification errors.

### 3.3 Episodic

Classification: Episodic

Each headline is processed independently.

The prediction for one headline does not change the prediction process for the next headline. The system does not maintain a sequence of decisions in which an earlier classification determines a later one.

Therefore, the task is episodic.

### 3.4 Static

Classification: Static during inference

Once a headline has been submitted, the input remains unchanged while the model processes it.

The application does not continuously monitor changes in the real-world news environment during an individual prediction.

Therefore, the inference environment is static.

### 3.5 Discrete

Classification: Discrete

The system works with text tokens and distinct classification categories.

The primary classification outputs are:

- Fake
- Real
- Insufficient Evidence

Although the model also produces a numerical confidence score, its main classification task is discrete.

### 3.6 Single-Agent

Classification: Single-Agent

One intelligent classification agent performs the prediction.

The user provides a headline, but does not compete or cooperate with another intelligent agent to determine the classification.

Therefore, the project is considered a single-agent system.

## 4. Task Environment Summary

| Property | Classification | Justification |
|---|---|---|
| Observability | Partially Observable | The actual truth and complete context of a news story are unavailable |
| Predictability | Deterministic at inference time | The same input and unchanged model produce repeatable predictions |
| Task Structure | Episodic | Each headline is classified independently |
| Environment Changes | Static | The submitted headline does not change during inference |
| State Representation | Discrete | The primary output consists of distinct categories |
| Number of Agents | Single-Agent | One classification agent performs the task |

## 5. Mapping PEAS to the Project Implementation

The PEAS framework describes the design of the agent. Its components are represented by different parts of the implementation.

| Component | Project Implementation |
|---|---|
| Sensors | Streamlit input and headline validation in HeadlineEnvironment |
| Feature Extraction | Existing TF-IDF vectorizer |
| Classification Agent | LogisticRegressionAgent |
| Prediction Processing | predict_headline() in src/inference.py |
| Actuators | Streamlit prediction, confidence, explanations, and warnings |
| Performance Evaluation | Existing model evaluation results and input-handling tests |
| Topic Exploration | get_cluster_summary() using saved clustering outputs |

## 6. Conclusion

The PEAS framework provides a structured description of the Fake News Headline Detector by identifying its performance measures, environment, actuators, and sensors.

The task environment analysis identifies the system as partially observable, deterministic at inference time, episodic, static, discrete, and single-agent.

These classifications describe the behavior and limitations of the implemented headline classification system. The detector provides predictions based on learned patterns in its training data; it does not independently verify the factual truth of news stories.
