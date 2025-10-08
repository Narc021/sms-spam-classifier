import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Page configuration
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .spam {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .not-spam {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        font-size: 18px;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize stemmer
ps = PorterStemmer()

# Text transformation function
@st.cache_data
def transform_text(text):
    """Preprocess text for spam detection"""
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Keep only alphanumeric
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Apply stemming
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

# Load models
@st.cache_resource
def load_models():
    """Load the trained models"""
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        return None, None

# Main UI
st.markdown("<h1 style='text-align: center; color: #667eea;'>🛡️ Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Protect yourself from spam messages using AI-powered detection</p>", unsafe_allow_html=True)

# Info section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    This tool uses **Machine Learning** to analyze your messages and determine if they're spam or legitimate.
    
    **Features:**
    - 🤖 Advanced NLP preprocessing
    - 📊 TF-IDF vectorization
    - 🎯 High accuracy spam detection
    - ⚡ Instant results
    
    **How to use:**
    1. Paste your message in the text area below
    2. Click the "Analyze Message" button
    3. Get instant results!
    """)

# Load models
tfidf, model = load_models()

if tfidf is not None and model is not None:
    # Input area
    st.markdown("---")
    input_sms = st.text_area(
        "📝 Enter your message",
        placeholder="Paste your email or SMS text here...",
        height=150,
        help="Enter the complete message you want to check"
    )
    
    # Statistics
    if input_sms:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", len(input_sms))
        with col2:
            st.metric("Words", len(input_sms.split()))
        with col3:
            st.metric("Lines", input_sms.count('\n') + 1)
    
    # Predict button
    if st.button('🔍 Analyze Message'):
        if input_sms.strip():
            with st.spinner('Analyzing message...'):
                # Preprocess
                transformed_sms = transform_text(input_sms)
                
                # Vectorize
                vector_input = tfidf.transform([transformed_sms])
                
                # Predict
                result = model.predict(vector_input)[0]
                prediction_proba = model.predict_proba(vector_input)[0]
                
                # Display results
                if result == 1:
                    st.markdown(f"""
                        <div class="prediction-box spam">
                            <h1>⚠️ SPAM DETECTED</h1>
                            <p style='font-size: 20px; margin-top: 1rem;'>This message appears to be spam</p>
                            <p style='font-size: 16px; opacity: 0.9;'>Confidence: {prediction_proba[1]*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("🚨 **Recommendation:** Be cautious with this message. Do not click on suspicious links or share personal information.")
                else:
                    st.markdown(f"""
                        <div class="prediction-box not-spam">
                            <h1>✅ LEGITIMATE MESSAGE</h1>
                            <p style='font-size: 20px; margin-top: 1rem;'>This message appears to be safe</p>
                            <p style='font-size: 16px; opacity: 0.9;'>Confidence: {prediction_proba[0]*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✨ This message looks legitimate, but always stay vigilant!")
                
                # Additional insights
                with st.expander("📊 View Analysis Details"):
                    st.write("**Processed Text:**")
                    st.code(transformed_sms[:200] + "..." if len(transformed_sms) > 200 else transformed_sms)
                    
                    st.write("**Prediction Probabilities:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Not Spam", f"{prediction_proba[0]*100:.2f}%")
                    with col2:
                        st.metric("Spam", f"{prediction_proba[1]*100:.2f}%")
        else:
            st.warning("⚠️ Please enter a message to analyze.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔒 Your privacy is protected. Messages are not stored or shared.</p>
        <p style='font-size: 14px; margin-top: 0.5rem;'>Made with ❤️ using Streamlit & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
