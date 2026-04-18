import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="AI Text Predictor", page_icon="✍️")

st.title("🚀 LSTM Next Word Predictor")
st.markdown("""
This system uses a **Stacked LSTM model** trained on Wikipedia data to predict the most likely next word in a sequence.
""")

# User input
seed_text = st.text_input("Enter a phrase to start:", "Artificial intelligence is")

if st.button("Predict Next Word"):
    if seed_text:
        with st.spinner('AI is thinking...'):
            try:
                # This connects to your FastAPI backend
                response = requests.post(
                     "localhost:8000/predict", 
                    
                    json={"seed_text": seed_text}
                )
                
                if response.status_status == 200:
                    result = response.json()
                    st.success(f"**Predicted Next Word:** {result['next_word']}")
                    
                    # Show the full sentence
                    st.info(f"**Full Sequence:** {seed_text} {result['next_word']}")
                else:
                    st.error("API is running but returned an error.")
            except:
                st.error("Could not connect to FastAPI. Make sure the server is running on port 8000!")
    else:
        st.warning("Please enter some text first.")