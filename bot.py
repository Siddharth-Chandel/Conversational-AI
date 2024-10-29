from keras.utils import pad_sequences
import streamlit as st
from keras.models import load_model
import pickle, numpy as np

MAX_LEN = 20

# Load encoder and decoder models
encoder_model = load_model("models/encoder_model.keras")
decoder_model = load_model("models/decoder_model.keras")
encoder_attention_model = load_model("models/encoder_attention_model.keras")
decoder_attention_model = load_model("models/decoder_attention_model.keras")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

def generate_response(input_text):
    # Encode the input as state vectors
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1 with only the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    # Loop to predict each word in the output sequence
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > MAX_LEN:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        # Update the target sequence and states
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Function to generate responses
def generate_response_attention(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')
    encoder_outs, state_h, state_c = encoder_attention_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c, attn = decoder_attention_model.predict([target_seq, encoder_outs, state_h, state_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > MAX_LEN:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        state_h, state_c = h, c

    return decoded_sentence.strip()

st.title("Chatbot with attention mechanism")
st.subheader("User")
user = st.text_input("1", label_visibility="collapsed", value="Hello, bot...")
bot_response = generate_response(user).replace(user,"").replace("Bot: ","").replace(':',"")
st.subheader(f"Bot [LSTM]: {generate_response(user)}")
st.subheader(f"Bot [LSTM + Attention mechanism]: {generate_response_attention(user)}")