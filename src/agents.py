from pathlib import Path
from difflib import get_close_matches
import re

import joblib

ROOT = Path(__file__).resolve().parents[1]

NAIVE_BAYES_MODEL_PATH = (
    ROOT / "outputs/cleaned_v2/reproducible_experiment/models/naive_bayes.joblib"
)

NAIVE_BAYES_VECTORIZER_PATH = (
    ROOT / "outputs/cleaned_v2/reproducible_experiment/models/tfidf_vectorizer.joblib"
)


class HeadlineEnvironment:
    """
    Represents the environment that receives and validates user headlines.
    """

    def __init__(self, headline=None):
        self.__headline = headline

    def set_headline(self, headline):
        self.__headline = headline

    def get_headline(self):
        return self.__headline

    def validate_headline(self):
        """
        Validate the input and normalize harmless whitespace.
        The original wording is not rewritten.
        """

        if not isinstance(self.__headline, str):
            return {
                "valid": False,
                "headline": None,
                "warnings": ["Headline must be a text string."]
            }

        cleaned_headline = " ".join(self.__headline.split())

        if not cleaned_headline:
            return {
                "valid": False,
                "headline": None,
                "warnings": ["Please enter a headline."]
            }

        warnings = []

        word_count = len(cleaned_headline.split())

        if word_count < 4:
            warnings.append(
                "This headline is very short. The prediction may be unreliable."
            )

        return {
            "valid": True,
            "headline": cleaned_headline,
            "warnings": warnings
        }


class TextClassifier:
    """
    Parent class for text classification agents.

    Stores a trained model and its matching TF-IDF vectorizer.
    """

    def __init__(self, model, vectorizer, model_name):
        self._model = model
        self._vectorizer = vectorizer
        self._model_name = model_name

    def _get_spelling_suggestions(self, headline):
        """
        Find possible spelling matches using the trained TF-IDF vocabulary.

        Suggestions are advisory only. The original headline is never changed.
        """

        vocabulary = self._vectorizer.vocabulary_

        words = re.findall(r"\b[A-Za-z]{4,}\b", headline)

        suggestions = []
        checked_words = set()

        for word in words:

            word_lower = word.lower()

            if word_lower in vocabulary or word_lower in checked_words:
                continue

            checked_words.add(word_lower)

            matches = get_close_matches(
                word_lower,
                vocabulary,
                n=2,
                cutoff=0.82
            )

            if matches:
                suggestions.append(
                    f"{word} → {' / '.join(matches)}"
                )

            if len(suggestions) == 5:
                break

        if suggestions:
            return [
                "Possible spelling matches (please verify; these are "
                "not automatic corrections): "
                + "; ".join(suggestions)
                + ". Names and unfamiliar terms may also be flagged."
            ]

        return []

    def _prepare_headline(self, headline):
        """
        Validate the headline and convert it into TF-IDF features.
        """

        environment = HeadlineEnvironment(headline)
        validation = environment.validate_headline()

        if not validation["valid"]:
            return validation, None

        cleaned_headline = validation["headline"]

        # Check for possible spelling mistakes.
        spelling_warnings = self._get_spelling_suggestions(
            cleaned_headline
        )

        validation["warnings"].extend(spelling_warnings)

        # Transform the original headline without correcting its words.
        features = self._vectorizer.transform([cleaned_headline])

        if features.nnz == 0:
            validation["valid"] = False
            validation["warnings"].append(
                "The headline contains no words recognized by the trained "
                "vocabulary. There is insufficient evidence for a reliable prediction."
            )
            return validation, None

        return validation, features

    def _create_result(self, label, confidence, warnings=None):
        """
        Create a consistent prediction result for all child agents.
        """

        return {
            "label": label,
            "confidence": confidence,
            "model": self._model_name,
            "top_words": [],
            "warnings": warnings if warnings is not None else []
        }

    def _get_class_label(self, class_value):
        """
        Convert numeric class labels into readable labels.
        """

        if class_value == 0:
            return "Fake"

        if class_value == 1:
            return "Real"

        return str(class_value)

    def predict(self, headline):
        """
        General prediction method.

        Child classes override this method to implement their
        model-specific prediction behavior.
        """

        validation, features = self._prepare_headline(headline)

        if not validation["valid"]:
            return self._create_result(
                "Insufficient evidence",
                None,
                validation["warnings"]
            )

        probabilities = self._model.predict_proba(features)[0]
        best_index = probabilities.argmax()

        predicted_class = self._model.classes_[best_index]
        confidence = float(probabilities[best_index])

        return self._create_result(
            self._get_class_label(predicted_class),
            confidence,
            validation["warnings"]
        )


class NaiveBayesAgent(TextClassifier):
    """
    Naive Bayes classification agent.
    Uses the saved Multinomial Naive Bayes model.
    """

    def __init__(self):
        if not NAIVE_BAYES_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Naive Bayes model not found: {NAIVE_BAYES_MODEL_PATH}"
            )

        if not NAIVE_BAYES_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Naive Bayes vectorizer not found: "
                f"{NAIVE_BAYES_VECTORIZER_PATH}"
            )

        model = joblib.load(NAIVE_BAYES_MODEL_PATH)
        vectorizer = joblib.load(NAIVE_BAYES_VECTORIZER_PATH)

        super().__init__(
            model,
            vectorizer,
            "Multinomial Naive Bayes"
        )

    def predict(self, headline):
        """
        Override the parent prediction method for Naive Bayes.
        """

        validation, features = self._prepare_headline(headline)

        if not validation["valid"]:
            return self._create_result(
                "Insufficient evidence",
                None,
                validation["warnings"]
            )

        probabilities = self._model.predict_proba(features)[0]

        predicted_class = self._model.predict(features)[0]
        class_index = list(self._model.classes_).index(predicted_class)

        confidence = float(probabilities[class_index])

        return self._create_result(
            self._get_class_label(predicted_class),
            confidence,
            validation["warnings"]
        )


class LogisticRegressionAgent(TextClassifier):
    """
    Logistic Regression classification agent.

    Uses the selected final model and its saved threshold.
    """

    def __init__(self):
        from src.explain import load_final_model

        model, vectorizer, threshold = load_final_model()

        super().__init__(
            model,
            vectorizer,
            "Logistic Regression"
        )

        self.__threshold = float(threshold)

    def get_threshold(self):
        return self.__threshold

    def predict(self, headline):
        """
        Override the parent prediction method using the
        project's selected Fake-class threshold.
        """

        validation, features = self._prepare_headline(headline)

        if not validation["valid"]:
            return self._create_result(
                "Insufficient evidence",
                None,
                validation["warnings"]
            )

        probabilities = self._model.predict_proba(features)[0]

        fake_index = list(self._model.classes_).index(0)
        real_index = list(self._model.classes_).index(1)

        fake_probability = float(probabilities[fake_index])
        real_probability = float(probabilities[real_index])

        if fake_probability >= self.__threshold:
            label = "Fake"
            confidence = fake_probability
        else:
            label = "Real"
            confidence = real_probability

        return self._create_result(
            label,
            confidence,
            validation["warnings"]
        )