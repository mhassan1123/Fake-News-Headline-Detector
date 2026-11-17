from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "outputs/cleaned_v2/models/final/final_model_bundle.joblib"


def load_final_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned-v2 final model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    return (
        bundle["model"],
        bundle["vectorizer"],
        bundle["threshold"]
    )


def get_influential_words(top_n=15):
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    model, vectorizer, threshold = load_final_model()

    if not hasattr(model, "coef_"):
        raise TypeError(
            "The selected model does not support coefficient explanations."
        )

    if len(model.classes_) != 2 or 1 not in model.classes_ or 0 not in model.classes_:
        raise ValueError(
            "This explanation function requires Fake (0) and Real (1) classes."
        )

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0].copy()

    real_index = list(model.classes_).index(1)

    # Orient coefficients so positive values support Real
    if real_index == 0:
        coefficients = -coefficients

    fake_indices = np.argsort(coefficients)[:top_n]
    real_indices = np.argsort(coefficients)[-top_n:][::-1]

    fake_words = [
        {
            "word": feature_names[i],
            "coefficient": float(coefficients[i])
        }
        for i in fake_indices
    ]

    real_words = [
        {
            "word": feature_names[i],
            "coefficient": float(coefficients[i])
        }
        for i in real_indices
    ]

    return {
        "fake_words": fake_words,
        "real_words": real_words
    }


def explain_headline(headline, top_n=5):
    if not isinstance(headline, str) or not headline.strip():
        raise ValueError("Please provide a non-empty headline.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    model, vectorizer, threshold = load_final_model()

    if not hasattr(model, "coef_"):
        raise TypeError(
            "The selected model does not support coefficient explanations."
        )

    if len(model.classes_) != 2 or 1 not in model.classes_ or 0 not in model.classes_:
        raise ValueError(
            "This explanation function requires Fake (0) and Real (1) classes."
        )

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0].copy()

    real_index = list(model.classes_).index(1)
    fake_index = list(model.classes_).index(0)

    # Orient coefficients so positive contributions support Real
    if real_index == 0:
        coefficients = -coefficients

    features = vectorizer.transform([headline])

    fake_probability = model.predict_proba(features)[0][fake_index]

    if fake_probability >= threshold:
        label = "Likely Fake"
        supporting_sign = -1
    else:
        label = "Likely Real"
        supporting_sign = 1

    word_indices = features.nonzero()[1]
    contributions = []

    for i in word_indices:
        tfidf_value = float(features[0, i])
        contribution = tfidf_value * float(coefficients[i])

        contributions.append({
            "word": feature_names[i],
            "tfidf": tfidf_value,
            "contribution": contribution
        })

    supporting = [
        item for item in contributions
        if item["contribution"] * supporting_sign > 0
    ]

    opposing = [
        item for item in contributions
        if item["contribution"] * supporting_sign < 0
    ]

    supporting.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True
    )

    opposing.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True
    )

    if features.nnz == 0:
        explanation_note = (
            "No words from the headline were found in the TF-IDF vocabulary. "
            "The prediction is based on the model's learned intercept, "
            "not identifiable headline words."
        )
    else:
        explanation_note = (
            "Word contributions show model influence, "
            "not whether the headline is factually true. "
            "The contributions do not include the model intercept."
        )

    return {
        "headline": headline,
        "prediction": label,
        "fake_probability": float(fake_probability),
        "threshold": float(threshold),
        "supporting_words": supporting[:top_n],
        "opposing_words": opposing[:top_n],
        "explanation_note": explanation_note
    }