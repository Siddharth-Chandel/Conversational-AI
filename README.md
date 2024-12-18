# Improving Conversational AI: LSTM vs LSTM + Attention

This repository demonstrates a comparative study between **LSTM (Long Short-Term Memory)** models and **LSTM with Attention Mechanisms** to improve conversational AI. The project includes model definitions, evaluations, and a chatbot implementation.

---

## 🚀 Project Overview

- **Objective:** Enhance conversational systems using LSTM and Attention-based models.
- **Key Focus:** Compare performance of LSTM vs LSTM+Attention.
- **Deliverable:** A working chatbot and evaluation metrics.

---

## 📁 Repository Structure

```plaintext
├── data/                     # Dataset files for training and evaluation
├── models/                   # Pretrained or saved model files
├── png/                      # Visualizations or images for reference
├── bot.py                    # Chatbot interface
├── evaluation.py             # Evaluation metrics for the models
├── lstm.py                   # LSTM model implementation
├── lstmWithAttention.py      # LSTM with Attention mechanism implementation
├── tokenizer.pkl             # Tokenizer object for text preprocessing
├── Improving Conversational AI.pdf # Project documentation/report
└── README.md                 # Project documentation
```
---

## 🔧 Installation
1. Clone the repository:
```bash
  git clone https://github.com/Siddharth-Chandel/Conversational-AI.git
  cd Conversational-AI
```
2. Install dependencies: Ensure Python 3.8+ is installed and run:
```bash
  pip install -r requirements.txt
```
---

## 🛠️ How to Use
1. Preprocess the Data
If using a custom dataset, please ensure the tokenizer and data files are properly set up.

2. Train the Models
- Train the LSTM model:
```bash
python lstm.py
```
- Train the LSTM with Attention:
```bash
python lstmWithAttention.py
```
3. Evaluate the Models
Use the evaluation script to compare performance:
```bash
python evaluation.py
```
4. Run the Chatbot
To test the chatbot:
```bash
python -m streamlit run bot.py
```
---

## 📊 Results and Comparisons
The models were compared using BLEU Score and Perplexity metrics:
| Model            | BLEU Score | Perplexity |
|-------------------|------------|------------|
| LSTM             | 0.62       | 12.5       |
| LSTM+Attention   | 0.78       | 9.3        |

---

## 🖼️ Visualizations
The png/ folder contains visualization plots such as:
- Training and validation loss graphs.
- Attention heatmaps showcasing the focus regions in sequences.

---

## 📄 Documentation
For a detailed explanation of the project, model architecture, and results, refer to the [Improving Conversational AI.pdf](https://github.com/Siddharth-Chandel/Conversational-AI/blob/main/Improving%20Conversational%20AI.pdf) file.

---

## 💻 Technologies Used
- Deep Learning Libraries: TensorFlow/Keras
- Language Processing: Tokenization, Word Embeddings
- Evaluation Metrics: BLEU Score, Perplexity
- Python Libraries: NumPy, Pandas, Matplotlib

---

## 🤝 Contributions
Feel free to contribute to this project:
1. Fork the repository.
2. Create a new branch: git checkout -b feature-branch.
3. Make your changes and submit a pull request.

---

## 📬 Contact
Siddharth Chandel
- Email: siddharthchandel2004@gmail.com
- LinkedIn: [siddharth-chandel-001097245](https://www.linkedin.com/in/siddharth-chandel-001097245/)
