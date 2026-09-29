import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


st.title("🎬 IMDb Sentiment Analyzer")

st.write(
    "Enter a movie review and the fine-tuned DistilBERT model "
    "will classify it as positive or negative."
)


review = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was absolutely amazing...",
    height=200,
)


if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():
        st.warning("Please enter a movie review.")

    else:
        try:

            response = requests.post(
                API_URL,
                json={"text": review},
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

            sentiment = result["sentiment"]
            confidence = result["confidence"]

            st.divider()

            if sentiment == "positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure the API is running."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")