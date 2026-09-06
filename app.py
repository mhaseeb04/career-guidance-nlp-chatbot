import streamlit as st
import pandas as pd 
import joblib
import string
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')

# Load the CSV saved from Jupyter
df = pd.read_csv('CareerGuidanceDataset.csv')

# Setup
st.set_page_config(page_title="Career Guidance Chatbot", layout="centered")
st.title('🎓 Career Guidance Chatbot')

# Career Categories 
career_roles = sorted(df['role'].unique())
selected_role = st.selectbox("🎯 Choose a Career Role", career_roles)

# load model
model = joblib.load('chatbot_model.pkl')

# load vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Function for cleaning the text / Preprocessing
def clean_text(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, '')
    return ' '.join([word for word in text.split() if word not in english_stopwords ])


# Form to ask the  question from the user
with st.form("chat_form"):
    user_ques = st.text_input('Ask me a career-related question: ')
    submitted = st.form_submit_button("🔍 Get Answer")


# Filter dataset for selected role
filtered_df = df[df['role'] == selected_role].copy()

# Clean the filtered questions
filtered_df['cleaned_question'] = filtered_df['question'].apply(clean_text)

# Vectorize the cleaned questions
filtered_dataset_ques = vectorizer.transform(filtered_df['cleaned_question'])


# Apply logic and give answer to user
if submitted and user_ques:
    
    # Clean the user question 
    clean_que = clean_text(user_ques)
    
    # Convert the cleaned user question into vectors
    user_question_vec = vectorizer.transform([clean_que])

    # Finding the similarity between the user question and questions in dataset
    similarity_scores = cosine_similarity(user_question_vec, filtered_dataset_ques)

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[0, best_index]

    if best_score > 0.3:
        st.success("Chatbot Answer: " + filtered_df['answer'].iloc[best_index])
    else:
        st.error("\nChatbot: Sorry, I don't have enough information about that.")
        
        
st.markdown(
    """
    <hr style="border: 1px solid #eaeaea; margin-top: 0px;">
    <div style="text-align: center; font-size: 15px;">
        👨‍💻 Developed by <b>Muhammad Haseeb</b><br>
    </div>
    """,
    unsafe_allow_html=True
)