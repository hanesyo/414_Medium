import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('postings.csv')
print(f"Total jobs: {len(df)}\n")

df = df.dropna(subset=['description'])
df['description_clean'] = df['description'].str.lower()
print(f"Jobs with description: {len(df)}\n")

tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, max_df=0.9)
job_vectors = tfidf.fit_transform(df['description_clean'])
print(f"TF-IDF features created: {job_vectors.shape}\n")

queries = {
    'ml_engineer': "machine learning model development Python TensorFlow PyTorch feature engineering neural network deep learning production algorithm scikit-learn optimization supervised learning",
    'analytics_engineer': "SQL data pipeline ETL dbt Airflow Spark data warehouse infrastructure analytics engineering transformation big data distributed systems cloud database",
    'product_analytics': "analytics A/B testing experimentation metrics product analytics Google Analytics tableau dashboard statistical analysis business intelligence KPI reporting stakeholder communication"
}

results_data = []

for career_name, query_text in queries.items():
    query_vector = tfidf.transform([query_text])
    similarities = cosine_similarity(query_vector, job_vectors).flatten()
    
    avg_sim = similarities.mean()
    max_sim = similarities.max()
    median_sim = np.median(similarities)
    
    print(f"{career_name.upper()}")
    print(f"Average: {avg_sim:.4f} | Median: {median_sim:.4f} | Max: {max_sim:.4f}\n")
    
    top_job_indices = np.argsort(similarities)[-10:][::-1]
    top_jobs = df.iloc[top_job_indices][['title', 'company_name']].copy()
    top_jobs['similarity'] = similarities[top_job_indices]
    top_jobs = top_jobs.reset_index(drop=True)
    top_jobs.index = top_jobs.index + 1
    
    print("Top 10:")
    print(top_jobs.to_string())
    print()
    
    results_data.append({
        'career_path': career_name,
        'avg_similarity': avg_sim,
        'median_similarity': median_sim,
        'max_similarity': max_sim
    })

print("SUMMARY")
summary_df = pd.DataFrame(results_data)
print(summary_df.to_string(index=False))
