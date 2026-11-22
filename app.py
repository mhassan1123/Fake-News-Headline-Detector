
import streamlit as st
import pandas as pd

from src.inference import predict_headline, get_cluster_summary


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Headline Detector",
    page_icon="📰",
    layout="wide"
)


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("📰 Fake News Headline Detector")

st.markdown(
    """
    Analyze news headlines using Machine Learning.

    This application uses TF-IDF and a trained Logistic Regression
    classifier to identify patterns associated with fake and real news.
    """
)

st.info(
    "Important: This application predicts headline patterns learned "
    "from its training data. It does not verify whether a news story "
    "is factually true."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("About the Model")

    st.write("**Selected Model:** Logistic Regression")

    st.write("**Text Representation:** TF-IDF")

    st.write("**Classification Threshold:** 0.477")

    st.write("**Clustering:** K-means")

    st.divider()

    st.subheader("Project Information")

    st.write(
        "This project compares machine learning approaches "
        "for fake news headline classification."
    )

    st.write(
        "The final prediction is generated using the selected "
        "Logistic Regression model."
    )


# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

analysis_tab, cluster_tab, about_tab = st.tabs(
    ["Analyze Headline", "Explore News Topics", "About Project"]
)


# ==================================================
# TAB 1: HEADLINE ANALYSIS
# ==================================================

with analysis_tab:

    st.subheader("Enter a News Headline")

    st.write(
        "Enter a headline below to receive a classification "
        "and an explanation of influential words."
    )

    with st.form("headline_form"):

        headline = st.text_area(
            "News headline",
            placeholder="Example: Government announces new economic policy",
            height=120
        )

        analyze_button = st.form_submit_button(
            "Analyze Headline",
            use_container_width=True
        )

    if analyze_button:

        with st.spinner("Analyzing headline..."):

            result = predict_headline(headline)

        st.session_state["prediction_result"] = result
        st.session_state["analyzed_headline"] = headline

    # Display the most recent prediction.
    if "prediction_result" in st.session_state:

        result = st.session_state["prediction_result"]

        st.divider()

        st.subheader("Analysis Result")

        st.write(
            "**Analyzed headline:**",
            st.session_state["analyzed_headline"]
        )

        label = result["label"]

        # Handle insufficient evidence.
        if label == "Insufficient evidence":

            st.warning(
                "Insufficient evidence: The system cannot make "
                "a reliable prediction for this headline."
            )

        elif label == "Prediction unavailable":

            st.error(
                "Prediction unavailable. Please check the model "
                "files and application logs."
            )

        else:

            if label == "Fake":
                st.error("Prediction: Likely Fake")
            else:
                st.success("Prediction: Likely Real")

            confidence = result["confidence"]

            # Prediction metrics.
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Fake Probability",
                    f"{result['fake_probability'] * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Classification Threshold",
                    f"{result['threshold']:.3f}"
                )

            st.caption(
                "Confidence represents the model's confidence "
                "in its classification, not the probability "
                "that the headline is factually true."
            )

            st.write("**Model Used:**", result["model"])

            # Influential words.
            st.divider()

            st.subheader("Influential Words")

            top_words = result["top_words"]

            if top_words:

                word_data = pd.DataFrame(top_words)

                word_data["contribution"] = word_data[
                    "contribution"
                ].round(4)

                st.dataframe(
                    word_data,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No identifiable influential words were "
                    "available for this prediction."
                )

            if result["explanation_note"]:

                st.caption(result["explanation_note"])

        # Display warnings.
        if result["warnings"]:

            st.divider()

            st.subheader("Input Warnings")

            for warning in result["warnings"]:
                st.warning(warning)


# ==================================================
# TAB 2: CLUSTER EXPLORER
# ==================================================

with cluster_tab:

    st.subheader("Explore News Topics")

    st.write(
        """
        The project uses K-means clustering to group headlines
        into 12 topic clusters.

        These clusters are generated from the existing training
        data and are not new predictions.
        """
    )

    try:

        cluster_data = get_cluster_summary()

        st.success(
            f"Successfully loaded {len(cluster_data)} existing clusters."
        )

        # Display cluster distribution.
        st.subheader("Cluster Distribution")

        chart_data = cluster_data.set_index("cluster")[
            ["headline_count"]
        ]

        st.bar_chart(chart_data)

        # Select a cluster.
        st.subheader("Explore Individual Clusters")

        selected_cluster = st.selectbox(
            "Select a cluster",
            cluster_data["cluster"].tolist()
        )

        selected_data = cluster_data[
            cluster_data["cluster"] == selected_cluster
        ].iloc[0]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Cluster Number",
                int(selected_data["cluster"])
            )

        with col2:

            st.metric(
                "Number of Headlines",
                f"{int(selected_data['headline_count']):,}"
            )

        st.write("**Frequent Topic Words**")

        st.write(selected_data["top_words"])

        st.write("**Example Headlines**")

        st.write(selected_data["example_headlines"])

    except Exception as error:

        st.error(
            "Unable to load the existing cluster summaries."
        )

        st.caption(str(error))


# ==================================================
# TAB 3: ABOUT PROJECT
# ==================================================

with about_tab:

    st.subheader("Project Overview")

    st.markdown(
        """
        ### Objective

        The objective is to classify news headlines as likely
        fake or real using machine learning.

        ### Technologies

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - Multinomial Naive Bayes
        - K-means Clustering
        - Streamlit

        ### Machine Learning Components

        **1. Text Classification**

        TF-IDF converts headline text into numerical features.
        Logistic Regression uses these features to classify
        headlines.

        **2. Model Comparison**

        Multinomial Naive Bayes and Logistic Regression are
        implemented as classification agents.

        **3. Topic Clustering**

        K-means groups headlines into clusters based on
        their TF-IDF representations.

        ### Limitations

        - The classifier learns patterns from its training data.
        - It does not access live news sources.
        - It does not fact-check events or verify claims.
        - Unusual wording, misspellings and incomplete headlines
          may reduce prediction reliability.
        - Model confidence should not be interpreted as factual certainty.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Fake News Headline Detector | Machine Learning Project"
)
