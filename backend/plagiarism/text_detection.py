import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download stopwords if not already downloaded
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# STOPWORDS
stop_words = set(stopwords.words('english'))

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess(text):
    if not text:
        return ""
    
    text = text.lower()
    
    # Remove punctuation
    for p in string.punctuation:
        text = text.replace(p, " ")
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    words = text.split()
    
    filtered = [
        word for word in words
        if word not in stop_words and len(word) > 1
    ]
    
    return " ".join(filtered)

# -----------------------------
# LCS ALGORITHM (Optimized)
# -----------------------------
def lcs(X, Y):
    if not X or not Y:
        return 0
    
    m, n = len(X), len(Y)
    
    # Optimize by using only two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev, curr = curr, prev
    
    return prev[n]

def lcs_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    
    length = lcs(text1, text2)
    max_len = max(len(text1), len(text2))
    
    similarity = (length / max_len) * 100 if max_len > 0 else 0
    return round(similarity, 2)

# -----------------------------
# TF-IDF + COSINE SIMILARITY
# -----------------------------
def tfidf_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    
    try:
        documents = [text1, text2]
        vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit features for performance
            stop_words='english'
        )
        matrix = vectorizer.fit_transform(documents)
        
        if matrix.shape[0] < 2:
            return 0
        
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])
        return round(float(similarity[0][0]) * 100, 2)
    except Exception as e:
        print(f"TF-IDF Error: {e}")
        return 0

# -----------------------------
# COMBINED SIMILARITY
# -----------------------------
def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    text1_clean = preprocess(text1)
    text2_clean = preprocess(text2)
    
    lcs_score = lcs_similarity(text1_clean, text2_clean)
    tfidf_score = tfidf_similarity(text1_clean, text2_clean)
    
    # Weighted score (give more weight to semantic similarity)
    final_score = (lcs_score * 0.3) + (tfidf_score * 0.7)
    
    return {
        "lcs_score": round(lcs_score, 2),
        "tfidf_score": round(tfidf_score, 2),
        "final_score": round(final_score, 2)
    }

def compare_with_repository(test_text, repository_documents):
    """
    Compare test document with all documents in repository
    Returns sorted results by similarity
    """
    results = []
    test_text_clean = preprocess(test_text)
    
    for doc in repository_documents:
        doc_text_clean = preprocess(doc['content'])
        
        scores = calculate_similarity(test_text_clean, doc_text_clean)
        
        results.append({
            "document_id": doc['id'],
            "title": doc['title'],
            "file_name": doc['file_name'],
            "lcs_score": scores['lcs_score'],
            "tfidf_score": scores['tfidf_score'],
            "final_score": scores['final_score'],
            "plagiarism_level": get_plagiarism_level(scores['final_score'])
        })
    
    # Sort by final score (highest first)
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return results

def get_plagiarism_level(score):
    if score >= 80:
        return "High Plagiarism"
    elif score >= 60:
        return "Moderate Plagiarism"
    elif score >= 40:
        return "Low Plagiarism"
    elif score >= 20:
        return "Minor Similarity"
    else:
        return "Original"