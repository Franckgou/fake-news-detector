import pandas as pd
import numpy as np
import re
import string
import nltk


# Add this at the beginning of your code, with the other nltk downloads
try:
    nltk.download('punkt_tab')
except:
    print("Note: punkt_tab download failed, but we can continue with regular punkt")

# First, download NLTK resources in a separate try-except block
try:
    print("Downloading NLTK resources...")
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    
# Only import the rest after NLTK setup
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from nltk.corpus import stopwords
    from collections import Counter
    from nltk.tokenize import word_tokenize
    
    print("All imports successful")
except Exception as e:
    print(f"Error importing libraries: {e}")

# Load the datasets
try:
    print("Loading datasets...")
    real_news = pd.read_csv(r"C:\Users\TAYO Franck\OneDrive\Documents\frontend\True.csv")
    fake_news = pd.read_csv(r"C:\Users\TAYO Franck\OneDrive\Documents\frontend\Fake.csv")
    
    # Add a label column to each dataset
    real_news['label'] = 1  # 1 for real news
    fake_news['label'] = 0  # 0 for fake news
    
    # Combine the datasets
    news_df = pd.concat([real_news, fake_news], ignore_index=True)
    
    # Display basic information
    print(f"Dataset shape: {news_df.shape}")
    print(f"Columns: {news_df.columns.tolist()}")
    
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")

# Simple data check before proceeding with visualization
try:
    # Display a few samples
    print("\nSample data (first 2 rows):")
    print(news_df.head(2))
    
    # Check for missing values
    print("\nMissing values:")
    print(news_df.isnull().sum())
    
    # Class distribution without visualization
    class_counts = news_df['label'].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    print("Basic checks completed successfully")
except Exception as e:
    print(f"Error performing basic checks: {e}")


# Add this after your basic checks code

# Visualization - Part 1: Class Distribution
try:
    print("Creating class distribution visualization...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=news_df, palette='viridis')
    plt.title('Distribution of Real vs Fake News')
    plt.xlabel('News Type (0=Fake, 1=Real)')
    plt.ylabel('Count')
    plt.savefig('class_distribution.png')  # Save the figure instead of showing it
    print("Class distribution visualization saved as 'class_distribution.png'")
except Exception as e:
    print(f"Error in visualization part 1: {e}")

# Visualization - Part 2: Text Length Analysis
try:
    print("Analyzing text length...")
    # Add text length as a feature
    news_df['text_length'] = news_df['text'].apply(len)
    
    # Descriptive statistics of text length by class
    print("\nText length statistics for REAL news:")
    print(news_df[news_df['label'] == 1]['text_length'].describe())
    
    print("\nText length statistics for FAKE news:")
    print(news_df[news_df['label'] == 0]['text_length'].describe())
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    sns.histplot(data=news_df, x='text_length', hue='label', bins=50, kde=True)
    plt.title('Distribution of Text Length by News Type')
    plt.xlabel('Text Length (characters)')
    plt.xlim(0, news_df['text_length'].quantile(0.99))  # Limit x-axis to exclude outliers
    plt.savefig('text_length_distribution.png')  # Save the figure
    print("Text length visualization saved as 'text_length_distribution.png'")
except Exception as e:
    print(f"Error in visualization part 2: {e}")

# Visualization - Part 3: Subject Distribution
try:
    print("Analyzing subject distribution...")
    if 'subject' in news_df.columns:
        # Show subject counts
        subject_counts = news_df.groupby(['subject', 'label']).size().unstack()
        print("\nSubject distribution by label:")
        print(subject_counts)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        subject_plot = sns.countplot(y='subject', hue='label', data=news_df, palette='viridis')
        plt.title('News Subjects by Real/Fake Label')
        plt.ylabel('Subject')
        plt.xlabel('Count')
        plt.legend(['Fake', 'Real'])
        plt.tight_layout()
        plt.savefig('subject_distribution.png')  # Save the figure
        print("Subject distribution visualization saved as 'subject_distribution.png'")
except Exception as e:
    print(f"Error in visualization part 3: {e}")

# Text Analysis - Most common words
try:
    print("Analyzing most common words...")
    
    # Function to get most common words
    def get_top_words(text_series, n=20):
        all_words = []
        stop_words = set(stopwords.words('english'))
        
        for text in text_series[:1000]:  # Use a subset to speed up processing
            try:
                words = word_tokenize(str(text).lower())
                words = [word for word in words if word.isalpha() and word not in stop_words]
                all_words.extend(words)
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
        
        return Counter(all_words).most_common(n)
    
    # Get top words for real and fake news
    print("\nMost common words in REAL news:")
    real_top_words = get_top_words(news_df[news_df['label'] == 1]['text'])
    print(real_top_words)
    
    print("\nMost common words in FAKE news:")
    fake_top_words = get_top_words(news_df[news_df['label'] == 0]['text'])
    print(fake_top_words)
    
    print("Word analysis completed")
except Exception as e:
    print(f"Error in word analysis: {e}")

# Text Preprocessing
try:
    print("\nStarting text preprocessing...")
    
    # Function to clean text
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Apply text cleaning to the 'text' column
    print("Cleaning text...")
    news_df['cleaned_text'] = news_df['text'].apply(clean_text)
    
    print("Sample of cleaned text:")
    print(news_df['cleaned_text'].iloc[0][:200] + "...")  # Print first 200 chars of first article
    
    print("Text preprocessing completed")
except Exception as e:
    print(f"Error in text preprocessing: {e}")

# Feature Extraction (Vectorization)
try:
    print("\nStarting feature extraction...")
    
    # Split the data into training and testing sets
    X = news_df['cleaned_text']
    y = news_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("Performing TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # Transform the test data
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    print("Feature extraction completed")
except Exception as e:
    print(f"Error in feature extraction: {e}")

# Model Training and Evaluation
try:
    print("\nStarting model training...")
    
    # Train a Logistic Regression model
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred_lr = lr_model.predict(X_test_tfidf)
    
    # Evaluate the model
    print("\nLogistic Regression Model Performance:")
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy: {lr_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_lr)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Logistic Regression')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix visualization saved as 'confusion_matrix.png'")
    
    # Train a Naive Bayes model for comparison
    print("\nTraining Multinomial Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred_nb = nb_model.predict(X_test_tfidf)
    
    # Evaluate the model
    print("\nNaive Bayes Model Performance:")
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    print(f"Accuracy: {nb_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nb))
    
    print("Model training and evaluation completed")
except Exception as e:
    print(f"Error in model training: {e}")

# Function to predict if a new article is real or fake
def predict_news(article_text, model=lr_model, vectorizer=tfidf_vectorizer):
    # Clean the text
    cleaned = clean_text(article_text)
    
    # Vectorize the text
    article_vector = vectorizer.transform([cleaned])
    
    # Make prediction
    prediction = model.predict(article_vector)[0]
    probability = model.predict_proba(article_vector)[0]
    
    # Return result
    if prediction == 1:
        return f"REAL NEWS (Confidence: {probability[1]:.2f})"
    else:
        return f"FAKE NEWS (Confidence: {probability[0]:.2f})"

# Test the function with a sample article
try:
    print("\nTesting model with sample articles...")
    
    # Test with a sample from the dataset
    sample_real = news_df[news_df['label'] == 1]['text'].iloc[0]
    print("\nSample real news prediction:")
    print(predict_news(sample_real))
    
    sample_fake = news_df[news_df['label'] == 0]['text'].iloc[0]
    print("\nSample fake news prediction:")
    print(predict_news(sample_fake))
    
    print("\nFake news detection model is ready to use!")
except Exception as e:
    print(f"Error in model testing: {e}")


# Save the trained model and vectorizer for use in the web app
import pickle

# Save the Logistic Regression model
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save sample articles for the web app
with open('sample_real_news.txt', 'w') as f:
    f.write(sample_real[:2000])  # Save the first 2000 characters

with open('sample_fake_news.txt', 'w') as f:
    f.write(sample_fake[:2000])  # Save the first 2000 characters

print("Model, vectorizer, and sample articles saved for web app")