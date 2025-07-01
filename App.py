import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import os
import re

# --- Set Streamlit page config FIRST ---
st.set_page_config(page_title="AI SMS/Email Spam Detector", layout="centered")

# --- Download required NLTK resources (safe check) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Initialize stemmer ---
ps = PorterStemmer()

# --- Text preprocessing function with advanced cleaning and error handling ---
def transform_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs and emails for privacy and better model focus
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.lower()
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    y = [i for i in tokens if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# --- Securely load vectorizer and model ---
def load_pickle_file(filename):
    if not os.path.exists(filename):
        st.error(f"Required file '{filename}' not found. Please ensure the model and vectorizer are present.")
        st.stop()
    try:
        # Only allow loading from current directory for security
        if not os.path.abspath(filename).startswith(os.getcwd()):
            st.error("Invalid file path for model/vectorizer.")
            st.stop()
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load '{filename}': {e}")
        st.stop()

tfidf = load_pickle_file('vectorizer.pkl')
model = load_pickle_file('model.pkl')

# --- Custom CSS for unique dark design and security ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #18181b 0%, #27272a 100%);
        color: #f4f4f5 !important;
    }
    .stTextArea textarea {
        background: #27272a;
        color: #f4f4f5;
        border-radius: 10px;
        font-size: 1.1em;
        border: 1.5px solid #6366f1;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #38bdf8 100%);
        color: #f4f4f5;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1em;
        margin-top: 10px;
        border: none;
        box-shadow: 0 2px 8px #6366f133;
        transition: 0.2s;
    }
    .stButton>button:hover {
        filter: brightness(1.15);
        transform: scale(1.03);
    }
    .result-box {
        padding: 1.5em;
        border-radius: 14px;
        margin-top: 1.5em;
        font-size: 1.25em;
        font-weight: bold;
        background: #23232b;
        border: 2.5px solid #818cf8;
        color: #f4f4f5;
        text-align: center;
        box-shadow: 0 2px 12px #818cf822;
        animation: fadeIn 0.7s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px);}
        to { opacity: 1; transform: translateY(0);}
    }
    .spam {
        background: #7f1d1d !important;
        border-color: #ef4444 !important;
        color: #fee2e2 !important;
    }
    .ham {
        background: #14532d !important;
        border-color: #22c55e !important;
        color: #dcfce7 !important;
    }
    .sidebar-content {
        color: #f4f4f5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
st.title("🤖 AI SMS/Email Spam Detector")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/spam.png", width=80)
    st.header("How it works")
    st.write(
        """
        - Enter or paste your SMS/email message below.
        - Click **Predict** to see if it's spam or not.
        - Try the sample messages for quick testing!
        """
    )
    st.markdown("---")
    st.subheader("Sample Messages")
    if st.button("Insert Spam Example", key="spam_sample"):
        st.session_state['input_sms'] = "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim your prize: http://fake-link.com"
    if st.button("Insert Ham Example", key="ham_sample"):
        st.session_state['input_sms'] = "Hi John, just wanted to remind you about our meeting tomorrow at 10 AM. Let me know if you need anything."

# --- Main input area with session state for better UX ---
if 'input_sms' not in st.session_state:
    st.session_state['input_sms'] = ""

input_sms = st.text_area(
    "Enter the message to classify",
    value=st.session_state['input_sms'],
    height=120,
    placeholder="Type or paste your SMS/email here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button('Predict', use_container_width=True)
with col2:
    clear_btn = st.button('Clear', use_container_width=True)

if clear_btn:
    st.session_state['input_sms'] = ""
    st.rerun()

if predict_btn:
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        if not transformed_sms:
            st.error("Message could not be processed. Please enter valid text.")
            st.stop()

        # 2. Vectorize
        try:
            vector_input = tfidf.transform([transformed_sms])
        except Exception as e:
            st.error(f"Vectorization failed: {e}")
            st.stop()

        # 3. Predict
        try:
            check_is_fitted(model)
            result = model.predict(vector_input)[0]
            # Defensive: check if predict_proba exists and is callable
            if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
                proba = model.predict_proba(vector_input)[0][result]
            else:
                proba = 0.0
        except NotFittedError:
            st.error("Model is not fitted. Please train it before use.")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # 4. Display result with confidence and unique style
        if result == 1:
            st.markdown(
                f"<div class='result-box spam'>🚫 <b>Spam</b><br>Confidence: {proba:.2%}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box ham'>✅ <b>Not Spam</b><br>Confidence: {proba:.2%}</div>",
                unsafe_allow_html=True
            )

st.markdown(
    "<div style='text-align:center; margin-top:2em; color:#64748b;'>"
    "Made with ❤️ using Streamlit & AI | Your data stays private."
    "</div>",
    unsafe_allow_html=True
)