import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to clean text (same as in your model training)
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        # Load the trained model from a pickle file
        with open('lr_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the vectorizer from a pickle file
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None

# Function to predict fake news
def predict_news(article_text, model, vectorizer):
    if not article_text.strip():
        return None, None
    
    # Clean the text
    cleaned = clean_text(article_text)
    
    # Vectorize the text
    article_vector = vectorizer.transform([cleaned])
    
    # Make prediction
    prediction = model.predict(article_vector)[0]
    probabilities = model.predict_proba(article_vector)[0]
    
    return prediction, probabilities

# Main function
def main():
    st.title("Fake News Detector")
    st.write("Upload or paste news article text to check if it's real or fake")
    
    # Load the model and vectorizer
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Paste Text", "Upload File"])
    
    with tab1:
        # Text input
        article_text = st.text_area("Paste the news article text here:", height=250)
        
        if st.button("Analyze Text"):
            if article_text:
                with st.spinner("Analyzing..."):
                    prediction, probabilities = predict_news(article_text, model, vectorizer)
                    display_results(prediction, probabilities, article_text)
            else:
                st.warning("Please enter some text to analyze")
    
    with tab2:
        # File upload
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        
        if uploaded_file is not None:
            article_text = uploaded_file.read().decode()
            st.text_area("File content:", article_text, height=250)
            
            if st.button("Analyze File"):
                with st.spinner("Analyzing..."):
                    prediction, probabilities = predict_news(article_text, model, vectorizer)
                    display_results(prediction, probabilities, article_text)

    # Add sample news for testing
    st.sidebar.header("Try with Sample News")
    if st.sidebar.button("Sample Real News"):
        # Load a sample real news article
        with open('sample_real_news.txt', 'r') as f:
            sample_real = f.read()
        st.text_area("Sample Real News:", sample_real, height=250)
        prediction, probabilities = predict_news(sample_real, model, vectorizer)
        display_results(prediction, probabilities, sample_real)
        
    if st.sidebar.button("Sample Fake News"):
        # Load a sample fake news article
        with open('sample_fake_news.txt', 'r') as f:
            sample_fake = f.read()
        st.text_area("Sample Fake News:", sample_fake, height=250)
        prediction, probabilities = predict_news(sample_fake, model, vectorizer)
        display_results(prediction, probabilities, sample_fake)

def display_results(prediction, probabilities, text):
    if prediction is None:
        st.warning("No text provided for analysis")
        return
    
    # Calculate confidence percentages
    real_prob = probabilities[1] * 100
    fake_prob = probabilities[0] * 100
    
    # Results container
    results_container = st.container()
    
    with results_container:
        # Prediction banner
        if prediction == 1:
            st.success(f"✅ This article appears to be REAL (Confidence: {real_prob:.2f}%)")
        else:
            st.error(f"⚠️ This article appears to be FAKE (Confidence: {fake_prob:.2f}%)")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = ['Fake News', 'Real News']
            values = [fake_prob, real_prob]
            ax.bar(labels, values, color=['#ff9999', '#66b3ff'])
            
            # Add percentage labels on top of the bars
            for i, v in enumerate(values):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
                
            ax.set_ylim(0, 100)
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Classification Confidence')
            st.pyplot(fig)
        
        with col2:
            # Donut chart
            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, texts, autotexts = ax.pie(
                [fake_prob, real_prob],
                labels=['Fake', 'Real'],
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'],
                startangle=90,
                wedgeprops=dict(width=0.5)
            )
            
            # Equal aspect ratio ensures the pie chart is circular
            ax.axis('equal')
            ax.set_title('Classification Result')
            st.pyplot(fig)
        
        # Add text analysis details
        st.subheader("Text Analysis Details")
        
        # Calculate basic text stats
        word_count = len(text.split())
        character_count = len(text)
        
        # Display stats
        st.text(f"Word Count: {word_count}")
        st.text(f"Character Count: {character_count}")
        
        # Display a sample of the cleaned text
        cleaned_text = clean_text(text)
        st.text_area("Cleaned text sample (first 500 characters):", 
                    cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
                    height=150)

if __name__ == "__main__":
    main()