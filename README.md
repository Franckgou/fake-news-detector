# Fake News Detector

A machine learning-based web application that detects fake news using Natural Language Processing.

## Features

- Real-time news article analysis
- Support for both text input and file upload
- Visualization of prediction results
- Sample articles for testing
- User-friendly interface built with Streamlit

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Franckgou/fake-news-detector.git
cd fake-news-detector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run fake_news_app.py
```

## Project Structure

- `fake_news_app.py`: Main Streamlit application
- `fake_news_detection.py`: Training script
- `requirements.txt`: Project dependencies
- `lr_model.pkl`: Trained model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer
