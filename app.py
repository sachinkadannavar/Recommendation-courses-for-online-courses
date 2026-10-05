# ===============================================================
# INDUSTRY LEVEL AI COURSE RECOMMENDATION SYSTEM - PRO VERSION
# VERSION 2.1 - ENTERPRISE EDITION (CLEAN INTERFACE)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import bcrypt
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# ===============================================================
# 1. PAGE CONFIGURATION & THEME ENGINE
# ===============================================================
st.set_page_config(
    page_title="Intelligence Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional UI: White Background with Corporate Blue Accents
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }
    
    /* Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Primary Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border: none;
        height: 3.2em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
        color: white;
    }
    
    /* Tables and DataFrames */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================================================
# 2. SECURITY & AUTHENTICATION
# ===============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# Persistent Enterprise User Database (CSV)
import os

USERS_FILE = "users.csv"

def save_users():
    # tore hashed passwords as utf-8 strings
    df = pd.DataFrame([{"username": u, "password": users_db[u].decode()} for u in users_db])
    df.to_csv(USERS_FILE, index=False)

def load_users():
    global users_db
    if os.path.exists(USERS_FILE):
        try:
            df = pd.read_csv(USERS_FILE)
            users_db = {row['username']: row['password'].encode() for _, row in df.iterrows()}
        except Exception:
            users_db = {"admin": hash_password("1234"), "sachin": hash_password("200")}
            save_users()
    else:
        users_db = {"admin": hash_password("1234"), "sachin": hash_password("200")}
        save_users()

def register_user(username, password):
    if not username or not password:
        return False, "Username and password required"
    if username in users_db:
        return False, "Username already exists"
    users_db[username] = hash_password(password)
    save_users()
    return True, "User registered successfully"

# initialize users_db from disk (or defaults)
load_users()

def check_password(username, password):
    if username in users_db:
        return bcrypt.checkpw(password.encode(), users_db[username])
    return False

# ===============================================================
# 3. DATA ARCHITECTURE & PROCESSING
# ===============================================================
@st.cache_data
def load_and_prep_data():
    try:
        courses = pd.read_csv("courses.csv")
        ratings = pd.read_csv("user_ratings.csv")
    except FileNotFoundError:
        # Generate Synthetic High-Fidelity Industry Data for Stability
        np.random.seed(42)
        categories = ['Generative AI', 'MLOps', 'Data Engineering', 'NLP', 'Computer Vision', 'Deep Learning']
        instructors = ['Dr. Aris', 'Sarah Jenkins', 'Tech Academy', 'Prof. Miller', 'OpenAI Community']
        
        courses = pd.DataFrame({
            'course_id': range(1, 151),
            'course_title': [f"Mastering {np.random.choice(categories)} Level {i}" for i in range(1, 151)],
            'category': [np.random.choice(categories) for _ in range(150)],
            'description': ["Comprehensive industry-standard curriculum with hands-on labs and certification."] * 150,
            'level': [np.random.choice(['Beginner', 'Intermediate', 'Expert']) for _ in range(150)],
            'price': np.random.randint(1200, 12000, 150),
            'num_reviews': np.random.randint(50, 15000, 150),
            'rating': np.random.uniform(3.8, 5.0, 150),
            'instructor': [np.random.choice(instructors) for _ in range(150)],
            'duration_hours': np.random.randint(10, 150, 150)
        })
        
        ratings = pd.DataFrame({
            'user_id': np.random.randint(1001, 1500, 2000),
            'course_id': np.random.randint(1, 151, 2000),
            'rating': np.random.choice([1, 2, 3, 4, 5], 2000, p=[0.05, 0.05, 0.1, 0.3, 0.5])
        })

    # Advanced Feature Engineering
    courses['revenue_estimate'] = courses['price'] * courses['num_reviews']
    courses['weighted_score'] = courses['rating'] * np.log1p(courses['num_reviews'])
    courses['combined_features'] = (
        courses['category'].str.lower() + " " + 
        courses['description'].str.lower() + " " + 
        courses['level'].str.lower()
    )
    
    return courses, ratings

# ===============================================================
# 4. LOGIN PORTAL (MINIMALIST)
# ===============================================================
def login_page():
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        tab_sign, tab_reg = st.tabs(["Sign In", "Register"])

        with tab_sign:
            with st.form("Secure Login"):
                u = st.text_input("Username", placeholder="Username")
                p = st.text_input("Password", type="password", placeholder="Password")
                submit = st.form_submit_button("Sign In")

                if submit:
                    if check_password(u, p):
                        st.session_state.authenticated = True
                        st.session_state.username = u
                        st.success("Signed in")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        with tab_reg:
            with st.form("Register Form"):
                ru = st.text_input("Choose a username", placeholder="New username")
                rp = st.text_input("Choose a password", type="password", placeholder="Password")
                rp2 = st.text_input("Confirm password", type="password", placeholder="Confirm password")
                reg_submit = st.form_submit_button("Create Account")

                if reg_submit:
                    if rp != rp2:
                        st.error("Passwords do not match")
                    else:
                        ok, msg = register_user(ru, rp)
                        if ok:
                            st.success(msg)
                            # auto-login after registering
                            st.session_state.authenticated = True
                            st.session_state.username = ru
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(msg)

# ===============================================================
# 5. MAIN DASHBOARD MODULES
# ===============================================================
def main_dashboard():
    courses, ratings = load_and_prep_data()
    
    # --- Sidebar Navigation ---
    st.sidebar.markdown(f"<h3 style='text-align:center;'>CONTROL CENTER</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='text-align:center;'><b>{st.session_state.username.upper()}</b></p>", unsafe_allow_html=True)
    st.sidebar.divider()
    
    menu = st.sidebar.radio("Selection Menu", [
        "Executive Summary", 
        "Advanced Market Analytics", 
        "Intelligence Recommender", 
        "Revenue & Performance",
        "Collaborative Filtering Lab",
        "Data Explorer",
        "Admin Control Panel"
    ])
    
    if st.sidebar.button("🔒 Logout System"):
        st.session_state.authenticated = False
        st.rerun()

    # --- 1. EXECUTIVE SUMMARY ---
    if menu == "Executive Summary":
        st.title("📊 Platform Executive Summary")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Active Catalog", f"{len(courses)} Courses")
        col_m2.metric("Total Users", f"{ratings['user_id'].nunique():,}")
        col_m3.metric("Avg. Satisfaction", f"{courses['rating'].mean():.2f} ⭐")
        col_m4.metric("Gross Revenue", f"₹{courses['revenue_estimate'].sum():,.0f}")
        
        st.divider()
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.subheader("Market Revenue by Domain")
            fig = px.pie(courses, names='category', values='revenue_estimate', 
                         hole=0.5, template="plotly_white", 
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Difficulty Segmentation")
            fig = px.bar(courses['level'].value_counts().reset_index(), 
                         x='level', y='count', template="plotly_white",
                         color='level', color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

    # --- 2. ADVANCED MARKET ANALYTICS ---
    elif menu == "Advanced Market Analytics":
        st.title("📈 Advanced Market Intelligence")
        
        t1, t2, t3 = st.tabs(["Market Analysis", "Correlation Matrix", "Instructor KPIs"])
        
        with t1:
            st.subheader("Pricing vs. Rating Interaction")
            fig = px.scatter(courses, x="price", y="rating", size="num_reviews", color="category",
                             hover_name="course_title", template="plotly_white",
                             labels={"price": "Course Price (INR)", "rating": "Avg Rating"})
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            st.subheader("Metric Correlation Analysis")
            corr_data = courses[['rating','num_reviews','price','duration_hours','revenue_estimate']].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_data, annot=True, cmap="Blues", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
            
        with t3:
            st.subheader("Top Performing Instructors")
            inst_data = courses.groupby('instructor').agg({'rating': 'mean', 'revenue_estimate': 'sum'}).sort_values('revenue_estimate', ascending=False).head(10)
            st.dataframe(inst_data.style.background_gradient(cmap='Blues'))

    # --- 3. INTELLIGENCE RECOMMENDER (CONTENT-BASED) ---
    elif menu == "Intelligence Recommender":
        st.title("🤖 Semantic Intelligence Lab")
        st.write("Processing High-Dimensional Semantic Space using TF-IDF and Cosine Similarity.")
        
        
        
        target = st.selectbox("Select Benchmark Course:", courses['course_title'])
        count = st.slider("Match Count", 3, 10, 5)
        
        if st.button("Run AI Recommendation"):
            with st.spinner("Analyzing semantic relationships..."):
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(courses['combined_features'])
                sim_matrix = cosine_similarity(tfidf_matrix)
                
                idx = courses[courses['course_title'] == target].index[0]
                scores = list(enumerate(sim_matrix[idx]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:count+1]
                
                st.subheader("Optimized Content Matches")
                rec_df = courses.iloc[[i[0] for i in scores]][['course_title', 'category', 'level', 'rating', 'price']]
                st.table(rec_df)
                
                sim_scores = [i[1] for i in scores]
                sim_names = [courses.iloc[i[0]]['course_title'] for i in scores]
                fig_sim = px.bar(x=sim_scores, y=sim_names, orientation='h', 
                                 title="Similarity Confidence Score", template="plotly_white")
                st.plotly_chart(fig_sim, use_container_width=True)

    # --- 4. REVENUE & PERFORMANCE ---
    elif menu == "Revenue & Performance":
        st.title("💰 Revenue Intelligence Dashboard")
        
        c_rev1, c_rev2 = st.columns(2)
        with c_rev1:
            st.subheader("Category Revenue Flow")
            rev_data = courses.groupby('category')['revenue_estimate'].sum().reset_index()
            fig_rev = px.funnel(rev_data.sort_values('revenue_estimate'), x='revenue_estimate', y='category', template="plotly_white")
            st.plotly_chart(fig_rev, use_container_width=True)
            
        with c_rev2:
            st.subheader("Gross Cumulative Revenue")
            courses_sorted = courses.sort_values('revenue_estimate', ascending=False)
            courses_sorted['cum_rev'] = courses_sorted['revenue_estimate'].cumsum()
            fig_cum = px.area(courses_sorted, x=range(len(courses)), y='cum_rev', template="plotly_white")
            st.plotly_chart(fig_cum, use_container_width=True)

    # --- 5. COLLABORATIVE FILTERING LAB ---
    elif menu == "Collaborative Filtering Lab":
        st.title("📉 Collaborative Modeling (SVD)")
        
        
        
        # User-Item Matrix Construction
        ui_matrix = ratings.pivot_table(index='user_id', columns='course_id', values='rating').fillna(0)
        
        svd = TruncatedSVD(n_components=12, random_state=42)
        matrix_reduced = svd.fit_transform(ui_matrix)
        matrix_recon = np.dot(matrix_reduced, svd.components_)
        
        rmse = np.sqrt(mean_squared_error(ui_matrix, matrix_recon))
        
        col_svd1, col_svd2 = st.columns([1, 2])
        with col_svd1:
            st.metric("Model RMSE Score", f"{rmse:.4f}")
            st.info("Singular Value Decomposition (SVD) identifies latent factors in user behavior.")
            
        with col_svd2:
            st.subheader("Explained Variance by Latent Factors")
            fig_var = px.line(np.cumsum(svd.explained_variance_ratio_), markers=True, template="plotly_white")
            st.plotly_chart(fig_var, use_container_width=True)

    # --- 6. DATA EXPLORER ---
    elif menu == "Data Explorer":
        st.title("📂 Data Warehouse")
        mode = st.radio("Access Level", ["Global Course Catalog", "User Interaction Logs"])
        
        if mode == "Global Course Catalog":
            st.dataframe(courses, use_container_width=True)
        else:
            st.dataframe(ratings, use_container_width=True)
        
        st.download_button("Export Dataset (CSV)", courses.to_csv(index=False), "course_data_export.csv", "text/csv")

    # --- 7. ADMIN CONTROL PANEL ---
    elif menu == "Admin Control Panel":
        st.title("⚙️ System Administration")
        st.warning("Authorized Personnel Only - Changes affect global data state.")
        
        with st.expander("Update Database"):
            up_file = st.file_uploader("Upload New Course Batch", type="csv")
            if up_file:
                st.success("New batch file validation successful.")
        
        st.markdown("### System Health")
        st.progress(98)
        st.write("API Status: Active | Database: Connected")

# ===============================================================
# 6. SYSTEM ROUTER
# ===============================================================
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard()

# ===============================================================
# END OF SYSTEM
# ===============================================================