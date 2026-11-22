# Learning Agent Architecture

## 1. Introduction

The Fake News Headline Detector follows the classical Learning Agent Architecture. It uses machine learning models to classify news headlines and provides explanations and warnings to users.

The architecture is explained through four components:

1. Performance Element
2. Learning Element
3. Critic
4. Problem Generator

The project performs model training and experimentation offline. The Streamlit application loads the saved final model and performs inference; it does not retrain the model when users submit headlines.

## 2. Performance Element

### Purpose

The Performance Element receives an input (percept), processes it, and selects an appropriate action.

### Implementation in this project

The deployed Performance Element uses the `LogisticRegressionAgent` class in `src/agents.py`.

The prediction workflow is:

1. Receive a headline from the user.
2. Validate the input.
3. Transform the headline using the saved TF-IDF vectorizer.
4. Obtain a prediction from the saved Logistic Regression model.
5. Apply the final decision threshold of 0.477.
6. Return the prediction and supporting information.

Possible outcomes include:

- Fake
- Real
- Insufficient Evidence

The inference engine in `src/inference.py` also provides confidence information, influential words, warnings, and other prediction details.

### Main implementation files

- `src/agents.py`
- `src/inference.py`
- `src/explain.py`
- `app.py`

## 3. Learning Element

### Purpose

The Learning Element improves the agent's performance by learning patterns from training data.

### Implementation in this project

Learning takes place during offline model development, not during normal use of the Streamlit application.

The project uses the following development scripts:

- `experiments/train_naive_bayes.py`
- `experiments/train_logistic_regression.py`
- `experiments/run_experiments.py`
- `experiments/finalize_model_selection.py`

The development process includes:

1. Preparing the training and validation data.
2. Representing headline text using TF-IDF features.
3. Training Naive Bayes and Logistic Regression models.
4. Comparing model performance.
5. Selecting the final model using validation results.
6. Saving the selected model and its associated vectorizer.

The reproducible experiment metadata records 53,006 training records, 13,252 validation records, and 10,000 TF-IDF features.

The final selected model is Logistic Regression with C = 2.0 and a decision threshold of 0.477.

The saved model bundle is located at:

`outputs/cleaned_v2/models/final/final_model_bundle.joblib`

**Important:** The deployed application loads this saved model. It does not automatically learn from new user headlines.

## 4. Critic

### Purpose

The Critic evaluates the agent's performance and provides feedback about how well it is performing.

### Implementation in this project

The Critic's role is represented by the model evaluation and auditing process.

The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- ROC-AUC
- Confusion Matrix

The validation results are used during model comparison and final model selection.

After selection, the final model is evaluated on the held-out test data. The final test evaluation is performed by:

`experiments/evaluate_final_model.py`

The reported final test results include:

- Macro F1-score: 0.8144
- ROC-AUC: 0.8972

The test evaluation is an assessment of the selected model. Test results are not used to retrain the model or tune its threshold.

### Main evaluation files

- `experiments/run_experiments.py`
- `experiments/finalize_model_selection.py`
- `experiments/evaluate_final_model.py`
- `outputs/cleaned_v2/audits/`
- `outputs/cleaned_v2/reproducible_experiment/results/`

## 5. Problem Generator

### Purpose

The Problem Generator suggests exploratory actions that can help the agent discover better ways to solve a problem.

For example, it may encourage trying alternative algorithms, features, or approaches to improve performance.

### Implementation in this project

The project does not implement a separate autonomous Problem Generator component.

However, during offline development, alternative approaches were explored through experiments. These included comparing models, evaluating TF-IDF configurations, and examining different decision thresholds.

These experiments represent development-time exploration. They should not be confused with an autonomous Problem Generator operating inside the deployed application.

## 6. Overall Architecture Workflow

The following workflow summarizes how the four components relate to this project:

1. The Learning Element trains and prepares machine learning models using training data.
2. The Critic evaluates the models using validation metrics and supports model selection.
3. The selected model is saved and loaded by the Performance Element.
4. The Performance Element processes user headlines and returns predictions.
5. The final model is evaluated on held-out test data to measure its performance.
6. Offline experiments explore alternative approaches, but a separate autonomous Problem Generator is not implemented.

## 7. Component Mapping

| Learning Agent Component | Project Implementation | Status |
|---|---|---|
| Performance Element | `LogisticRegressionAgent` and inference engine | Implemented |
| Learning Element | Offline TF-IDF and model-training pipeline | Implemented during development |
| Critic | Validation and test evaluation | Evaluation implemented |
| Problem Generator | No separate autonomous component | Not explicitly implemented |

## 8. Conclusion

The Fake News Headline Detector demonstrates the main ideas of a Learning Agent Architecture through offline machine learning, model evaluation, and real-time inference.

The Learning Element prepares the models, the Critic evaluates their performance, and the Performance Element uses the selected model to classify user-provided headlines.

Although alternative approaches were explored during development, the project does not contain a separate autonomous Problem Generator. Clearly identifying this limitation provides an accurate representation of the implemented system.
