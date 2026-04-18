# 🧠 LSTM-Based Sequence Prediction System with Deployment

A deep learning project that trains an LSTM language model on Wikipedia articles and deploys it as a production-ready **FastAPI** REST API. Given a seed phrase, the model predicts the next most likely word using temperature-controlled sampling.

---

## 👥 Group Details

| Name | Roll Number |
|---|---|
| Darshan Bhabad | 202301040169 |
| Mitesh Chaudhari | 202301040106 |
| Krishna Tolani | 2023010400 |

---

## 🔗 Links

- 📦 **GitHub Repo:** [LSTM-Based-Sequence-Prediction-System-with-Deployment](https://github.com/DarshanBhabad/LSTM-Based-Sequence-Prediction-System-with-Deployment)
- 📓 **Kaggle Notebook:** [lstm-based-sequence-prediction-system](https://www.kaggle.com/code/bboyattitude/lstm-based-sequence-prediction-system)

---

## 📌 Project Overview

This project is divided into two parts:

1. **Model Training (Notebook)** — Fetches Wikipedia articles, preprocesses the text, trains a stacked LSTM model, and exports the model artifacts.
2. **Deployment (FastAPI)** — Loads the trained model and exposes a REST API endpoint for real-time next-word prediction.

---

## 📂 Project Structure

```
├── lstm-based-sequence-prediction-system.ipynb  # Training notebook
├── main.py                                      # FastAPI deployment server
├── model/
│   ├── lstm_text_model.h5                       # Trained LSTM model
│   ├── tokenizer.pkl                            # Fitted Keras tokenizer
│   └── config.pkl                               # Sequence config (max_seq_len)
└── README.md
```

---

## 📊 Dataset Details

| Property | Details |
|---|---|
| **Source** | Wikipedia API (MediaWiki) |
| **Python Wrapper** | `wikipedia-api` |
| **Topics Fetched** | Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks |

### Preprocessing Steps

1. Lowercasing all text
2. Removing Wikipedia-specific headers (e.g., "See Also", "References")
3. Regex cleaning to strip special characters and citations (e.g., `[1]`, `[22]`)
4. Tokenization using Keras `Tokenizer`
5. N-gram sequence creation with a sliding window of 10 words
6. Pre-padding all sequences to uniform length

---

## 🏗️ Model Architecture

```
Layer (type)              Output Shape         
─────────────────────────────────────────────────
Embedding                 (None, seq_len, 100)    
LSTM (return_sequences)   (None, seq_len, 150)    
Dropout (0.2)             (None, seq_len, 150)    
LSTM                      (None, 100)             
Dense (softmax)           (None, total_words)     
─────────────────────────────────────────────────
```

| Parameter | Value |
|---|---|
| Embedding Dimensions | 100 |
| LSTM Units (Layer 1) | 150 |
| LSTM Units (Layer 2) | 100 |
| Dropout Rate | 0.2 |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Epochs | 50 |
| Batch Size | 64 |

---

## ⚙️ Prediction Logic

The API uses **temperature scaling** with **weighted random sampling** for predictions — instead of always returning the top-1 word, it samples from the probability distribution to produce more natural and varied outputs.

```
logits = log(probabilities + ε) / temperature
probabilities = softmax(logits)
predicted_word = multinomial_sample(probabilities)
```

| Temperature | Behaviour |
|---|---|
| `< 1.0` | More confident and repetitive predictions |
| `= 1.0` | Default — balanced predictions |
| `> 1.0` | More diverse and creative predictions |

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/DarshanBhabad/LSTM-Based-Sequence-Prediction-System-with-Deployment.git
cd LSTM-Based-Sequence-Prediction-System-with-Deployment
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn tensorflow numpy wikipedia-api
```

### 3. Train the Model

Run the Jupyter notebook to train the model and export the required files into the `model/` directory:

```bash
jupyter notebook lstm-based-sequence-prediction-system.ipynb
```

### 4. Run the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

---

## 📡 API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "LSTM Text Prediction API is Running!"
}
```

---

### `POST /predict`
Predicts the next word for a given seed text.

**Request Body:**
```json
{
  "seed_text": "artificial intelligence is",
  "temperature": 1.0
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `seed_text` | `string` | ✅ | — | The input phrase to predict from |
| `temperature` | `float` | ❌ | `1.0` | Controls prediction creativity |

**Response:**
```json
{
  "input": "artificial intelligence is",
  "next_word": "used"
}
```

---

## 🧪 Example Usage (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"seed_text": "machine learning is", "temperature": 0.8}'
```

**Interactive Docs:** Visit `http://127.0.0.1:8000/docs` for the auto-generated Swagger UI.

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| API Framework | FastAPI |
| Data Source | Wikipedia API (`wikipedia-api`) |
| Serialization | Pickle |
| Server | Uvicorn |

---

## 📜 License

This project was developed as an academic group project. Feel free to fork and build upon it.
