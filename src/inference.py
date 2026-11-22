
from pathlib import Path
import logging

import pandas as pd

from src.agents import LogisticRegressionAgent
from src.explain import explain_headline


ROOT = Path(__file__).resolve().parents[1]

CLUSTER_SUMMARY_PATH = (
    ROOT / "outputs/cleaned_v2/clustering/cluster_summary.csv"
)

logger = logging.getLogger(__name__)


def predict_headline(headline):
    """
    Predict whether a headline is likely Fake or Real.

    Uses the selected Logistic Regression agent and the existing
    explanation function.

    Returns a dictionary containing:
    label, confidence, model, top_words, warnings,
    fake_probability, threshold and explanation_note.
    """

    try:
        agent = LogisticRegressionAgent()

        # Generate the classification result.
        prediction = agent.predict(headline)

        result = {
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "model": prediction["model"],
            "top_words": [],
            "warnings": list(prediction["warnings"]),
            "fake_probability": None,
            "threshold": agent.get_threshold(),
            "explanation_note": None
        }

        # Do not generate explanations for invalid or unknown inputs.
        if prediction["label"] == "Insufficient evidence":
            result["explanation_note"] = (
                "The headline does not contain enough recognized "
                "information for a reliable prediction."
            )
            return result

        # Reuse the project's existing explanation functionality.
        explanation = explain_headline(headline, top_n=5)

        result["fake_probability"] = explanation["fake_probability"]
        result["threshold"] = explanation["threshold"]
        result["explanation_note"] = explanation["explanation_note"]

        # Combine words supporting and opposing the prediction.
        for item in explanation["supporting_words"]:
            result["top_words"].append({
                "word": item["word"],
                "contribution": item["contribution"],
                "direction": "Supports prediction"
            })

        for item in explanation["opposing_words"]:
            result["top_words"].append({
                "word": item["word"],
                "contribution": item["contribution"],
                "direction": "Opposes prediction"
            })

        # Show the words with the strongest absolute contributions first.
        result["top_words"].sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True
        )

        return result

    except Exception:
        logger.exception("Headline prediction failed.")

        return {
            "label": "Prediction unavailable",
            "confidence": None,
            "model": "Logistic Regression",
            "top_words": [],
            "warnings": [
                "The prediction could not be generated. "
                "Please check the model files and application logs."
            ],
            "fake_probability": None,
            "threshold": None,
            "explanation_note": None
        }


def get_cluster_summary():
    """
    Load the previously generated K-means cluster summaries.

    This function does not retrain the clustering model.

    Returns:
        pandas.DataFrame: Existing cluster information.
    """

    if not CLUSTER_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Cluster summary file not found: {CLUSTER_SUMMARY_PATH}"
        )

    cluster_data = pd.read_csv(CLUSTER_SUMMARY_PATH)

    required_columns = {
        "cluster",
        "headline_count",
        "top_words",
        "example_headlines"
    }

    missing_columns = required_columns.difference(cluster_data.columns)

    if missing_columns:
        raise ValueError(
            "Cluster summary is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if cluster_data.empty:
        raise ValueError("The cluster summary file contains no records.")

    return cluster_data
