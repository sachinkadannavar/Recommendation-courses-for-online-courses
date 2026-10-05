"""
ONLINE COURSE RECOMMENDATION SYSTEM
-----------------------------------
Includes:
1. Data Cleaning
2. Feature Engineering
3. 12 Advanced Visualizations
4. Content-Based Filtering
5. Collaborative Filtering (SVD)
6. Hybrid Recommendation
7. Model Evaluation
"""

# ================================
# 1. IMPORT LIBRARIES
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)


# ================================
# 2. LOAD DATA
# ================================

courses = pd.read_csv("courses.csv")
ratings = pd.read_csv("user_ratings.csv")

print("Courses Shape:", courses.shape)
print("Ratings Shape:", ratings.shape)


# ================================
# 3. DATA CLEANING
# ================================

# Remove duplicates
courses.drop_duplicates(inplace=True)

# Fill missing values
courses['description'] = courses['description'].fillna('')
courses['category'] = courses['category'].fillna('unknown')
courses['level'] = courses['level'].fillna('unknown')

# Convert text to lowercase
courses['category'] = courses['category'].str.lower()
courses['level'] = courses['level'].str.lower()

# Weighted score
courses['weighted_score'] = courses['rating'] * np.log1p(courses['num_reviews'])

# Revenue estimate
courses['revenue_estimate'] = courses['price'] * courses['num_reviews']


# ================================
# 4. EXPLORATORY DATA ANALYSIS
# ================================

# 1 Category Distribution
sns.countplot(y='category', data=courses,
              order=courses['category'].value_counts().index)
plt.title("Course Category Distribution")
plt.show()

# 2 Rating Distribution
sns.histplot(courses['rating'], bins=20, kde=True)
plt.title("Rating Distribution")
plt.show()

# 3 Price vs Rating
sns.scatterplot(x='price', y='rating', data=courses)
plt.title("Price vs Rating")
plt.show()

# 4 Duration Distribution
sns.boxplot(x=courses['duration_hours'])
plt.title("Course Duration Boxplot")
plt.show()

# 5 Level Distribution
sns.countplot(x='level', data=courses)
plt.title("Course Level Distribution")
plt.show()

# 6 Correlation Heatmap
sns.heatmap(courses[['rating','num_reviews','price','duration_hours']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 7 Top 10 Reviewed Courses
top_reviews = courses.sort_values('num_reviews', ascending=False).head(10)
sns.barplot(x='num_reviews', y='course_title', data=top_reviews)
plt.title("Top 10 Most Reviewed Courses")
plt.show()

# 8 Top Revenue Courses
top_revenue = courses.sort_values('revenue_estimate', ascending=False).head(10)
sns.barplot(x='revenue_estimate', y='course_title', data=top_revenue)
plt.title("Top Revenue Generating Courses")
plt.show()

# 9 Avg Rating by Category
cat_rating = courses.groupby('category')['rating'].mean().sort_values()
cat_rating.plot(kind='barh')
plt.title("Average Rating by Category")
plt.show()

# 10 Price by Level
sns.boxplot(x='level', y='price', data=courses)
plt.title("Price Distribution by Level")
plt.show()

# 11 Reviews vs Rating Trend
sns.regplot(x='num_reviews', y='rating', data=courses)
plt.title("Reviews vs Rating Trend")
plt.show()

# 12 Rating vs Duration
sns.scatterplot(x='duration_hours', y='rating', data=courses)
plt.title("Duration vs Rating")
plt.show()


# ================================
# 5. CONTENT-BASED FILTERING
# ================================

courses['combined_features'] = (
    courses['category'] + " " +
    courses['description'] + " " +
    courses['level']
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(courses['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_recommendation(title, top_n=5):
    if title not in courses['course_title'].values:
        return "Course not found."
    
    idx = courses[courses['course_title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores,
                               key=lambda x: x[1],
                               reverse=True)[1:top_n+1]
    
    indices = [i[0] for i in similarity_scores]
    return courses[['course_title','rating','price']].iloc[indices]


# ================================
# 6. COLLABORATIVE FILTERING (SVD)
# ================================

user_course_matrix = ratings.pivot_table(
    index='user_id',
    columns='course_id',
    values='rating'
).fillna(0)

svd = TruncatedSVD(n_components=50, random_state=42)
matrix_reduced = svd.fit_transform(user_course_matrix)
matrix_reconstructed = np.dot(matrix_reduced, svd.components_)

rmse = np.sqrt(mean_squared_error(user_course_matrix,
                                  matrix_reconstructed))
print("Collaborative Filtering RMSE:", rmse)


def collaborative_recommendation(user_id, top_n=5):
    if user_id not in user_course_matrix.index:
        return "User not found."
    
    user_index = list(user_course_matrix.index).index(user_id)
    predicted_ratings = matrix_reconstructed[user_index]
    
    top_indices = np.argsort(predicted_ratings)[::-1][:top_n]
    recommended_courses = user_course_matrix.columns[top_indices]
    
    return courses[courses['course_id'].isin(recommended_courses)][
        ['course_title','rating','price']
    ]


# ================================
# 7. HYBRID RECOMMENDATION
# ================================

def hybrid_recommendation(title, user_id, top_n=5):
    content_rec = content_recommendation(title, top_n)
    collab_rec = collaborative_recommendation(user_id, top_n)
    
    print("----- Content-Based Recommendations -----")
    print(content_rec)
    
    print("\n----- Collaborative Filtering Recommendations -----")
    print(collab_rec)


# ================================
# 8. SAMPLE EXECUTION
# ================================

print("\nSample Content-Based Recommendation:\n")
print(content_recommendation(courses['course_title'].iloc[0]))

print("\nSample Collaborative Recommendation:\n")
print(collaborative_recommendation(ratings['user_id'].iloc[0]))


# ================================
# 9. BUSINESS INSIGHTS
# ================================

print("\nTop 5 Revenue Categories:")
print(courses.groupby('category')['revenue_estimate']
      .sum()
      .sort_values(ascending=False)
      .head())

print("\nBest Rated Courses:")
print(courses.sort_values('rating', ascending=False)
      .head(5)[['course_title','rating']])

print("\nProject Completed Successfully 🚀")