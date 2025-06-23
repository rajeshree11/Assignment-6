# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob

# Load spaCy English model safely
try:
    import en_core_web_sm
    nlp = en_core_web_sm.load()
except:
    nlp = spacy.blank("en")
    st.warning("⚠️ Fallback to blank SpaCy model. Run `python -m spacy download en_core_web_sm` to enable full entity recognition.")

# Streamlit App Config
st.set_page_config(page_title="📰 News Headline Analyzer", layout="wide")
st.title("📰 News Headline Analyzer")
st.markdown("""
Upload a CSV or GZ file containing **news headlines** to analyze their sentiment and extract named entities
such as people, organizations, and locations.
""")

# File Uploader
uploaded_file = st.file_uploader("📁 Upload News Headlines Dataset", type=["csv", "gz"])

if uploaded_file:
    if uploaded_file.name.endswith(".gz"):
        df = pd.read_csv(uploaded_file, compression="gzip")
    else:
        df = pd.read_csv(uploaded_file)

    # Detect headline column
    possible_cols = [col for col in df.columns if "title" in col.lower() or "headline" in col.lower()]
    if possible_cols:
        text_column = possible_cols[0]
    else:
        text_column = st.selectbox("Select the column with news headlines", df.columns)

    df = df.dropna(subset=[text_column]).reset_index(drop=True)

    if st.button("🧠 Analyze Headlines"):
        sentiment_scores = []
        entities_list = []

        for line in df[text_column]:
            # Sentiment analysis
            blob = TextBlob(line)
            sentiment_scores.append(blob.sentiment.polarity)

            # Named entity recognition
            doc = nlp(line)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            entities_list.append(entities)

        # Store results
        df["Sentiment"] = sentiment_scores
        df["Entities"] = entities_list

        st.success("✅ Analysis complete!")
        st.write("### 📊 Processed Headlines Preview")
        st.dataframe(df[[text_column, "Sentiment", "Entities"]])

        # Download button
        csv_result = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv_result, "news_headlines_nlp_results.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("🔍 This app is ideal for analyzing media tone, tracking public sentiment, or researching political narratives.")
