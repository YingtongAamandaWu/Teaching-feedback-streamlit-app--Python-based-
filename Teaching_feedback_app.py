#!/usr/bin/env python
# coding: utf-8

# Some helpful tutorials:
# 
# https://medium.com/@iamleonie/recreating-amazons-new-generative-ai-feature-product-review-summaries-50640e40872a
# 
# https://medium.com/@abed63/flask-application-with-openai-chatgpt-integration-tutorial-958588ac6bdf
# 
# https://github.com/openai/openai-python/discussions/742
# 
# https://www.listendata.com/2023/03/open-source-chatgpt-models-step-by-step.html
# 
# https://www.geeksforgeeks.org/mastering-text-summarization-with-sumy-a-python-library-overview/
# 
# https://blog.streamlit.io/host-your-streamlit-app-for-free/
# 

# In[2]:


import streamlit as st
from textblob import TextBlob
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.tokenize import sent_tokenize
import nltk
from textblob import download_corpora

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)  # Tokenizer for sentence splitting
nltk.download('stopwords', quiet=True)  # Stopwords for filtering

# Download TextBlob corpora
from textblob.download_corpora import download_all
download_all()  # Correct function to download all required corpora


# In[ ]:


# Ensure necessary resources are downloaded
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # For wordnet synonyms
nltk.download('brown', quiet=True)    # Optional for TextBlob corpora

# Function to generate summary using Sumy
def summarize_text_sumy(text, algorithm="LSA", sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    if algorithm == "LSA":
        summarizer = LsaSummarizer()
    elif algorithm == "Luhn":
        summarizer = LuhnSummarizer()
    elif algorithm == "TextRank":
        summarizer = TextRankSummarizer()
    elif algorithm == "LexRank":
        summarizer = LexRankSummarizer()
    else:
        summarizer = LsaSummarizer()  # Default

    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

# Streamlit App
st.title("Feedback Summarizer with NLP (Natural Language Processing), Sentiment Analysis, and Review Summary")
st.write(
    "Please center text below, select a summarization algorithm, and the app will provide "
    "a summary, sentiment analysis, an interactive sentiment boxplot, and a WordCloud visualization."
)

# Input Text
input_text = st.text_area("Enter your text here:", height=150)

# Dropdown Menu for Summarization Algorithm Selection
summarization_algorithm = st.selectbox(
    "Select a text summarization algorithm:",
    ["LSA", "Luhn", "TextRank", "LexRank"],
    index=0,
)

# Input for Exclusion Words
excluded_words_input = st.text_input(
    "Enter additional words to exclude from the WordCloud (comma-separated):", 
    placeholder="e.g., data, analysis"
)

# Process Excluded Words
user_excluded_words = [
    word.strip().lower() for word in excluded_words_input.split(",") if word.strip()
]

# Combine user exclusions with NLTK's stopwords
stop_words = set(stopwords.words('english')).union(user_excluded_words)

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Sentiment Analysis
        blob = TextBlob(input_text)
        sentences = blob.sentences
        polarities = [TextBlob(str(sentence)).sentiment.polarity for sentence in sentences]

        # Word and Sentence Count
        word_count = len(blob.words)
        sentence_count = len(sentences)

        # Generate Summary using Sumy
        summarized_text = summarize_text_sumy(input_text, algorithm=summarization_algorithm)

        # Display Results
        st.subheader("Results")
        avg_sentiment = sum(polarities) / len(polarities) if polarities else 0
        sentiment_result = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
        st.markdown(f"**Average Sentiment:** {sentiment_result} ({avg_sentiment:.2f})")
        st.markdown(f"**Word Count:** {word_count}")
        st.markdown(f"**Sentence Count:** {sentence_count}")
        st.markdown(f"**Summary Algorithm:** {summarization_algorithm}")

        st.subheader("Summary")
        st.write(summarized_text)

        # Boxplot for Sentiment Polarity
        if polarities:
            st.subheader("Sentiment Polarity Distribution")
            data = {"Sentence": [str(sentence) for sentence in sentences], "Polarity": polarities}
            fig = px.box(data, y="Polarity", points="all", hover_data=["Sentence"])
            st.plotly_chart(fig, use_container_width=True)

        # Generate and Display WordCloud (With Stopwords and Exclusions)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stop_words,
        ).generate(input_text)
        st.subheader("WordCloud")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

