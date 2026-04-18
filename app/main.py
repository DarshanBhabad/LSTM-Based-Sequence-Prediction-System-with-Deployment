from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware # Added for safety

# 1. Initialize FastAPI app
app = FastAPI(title="LSTM Next Word Predictor")

# Add CORS so the API can be accessed by other tools/frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Model and Assets
model = tf.keras.models.load_model('model/lstm_text_model.h5')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('model/config.pkl', 'rb') as f:
    config = pickle.load(f)

max_sequence_len = config['max_seq_len']

# Optimization: Create a reverse dictionary once at startup for O(1) lookup
# This is much faster than looping through the tokenizer every time
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# 3. Request Body Schema
class TextRequest(BaseModel):
    seed_text: str
    temperature: float = 1.0  # Optional: User can now control "creativity"

# 4. Improved Prediction Logic (Sampling + Temperature)
def get_prediction(seed_text, temperature=1.0):
    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Get the raw probability distribution
    preds = model.predict(token_list, verbose=0)[0]
    
    # --- TEMPERATURE SCALING ---
    # We apply log to probabilities, divide by temperature, then re-exponentiate
    # T > 1.0 makes results more diverse; T < 1.0 makes them more confident
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    # --- WEIGHTED SAMPLING ---
    # Instead of picking the #1 index, we pick based on the distribution
    probas = np.random.multinomial(1, preds, 1)
    predicted_index = np.argmax(probas)
    
    # Quick lookup from our optimized dictionary
    return index_to_word.get(predicted_index, "")

# 5. API Endpoints
@app.get("/")
def home():
    return {"message": "LSTM Text Prediction API is Running!"}

@app.post("/predict")
def predict(request: TextRequest):
    prediction = get_prediction(request.seed_text, request.temperature)
    return {
        "input": request.seed_text,
        "next_word": prediction
    }