import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
import pickle

# Load the dataset
df = pd.read_csv("data/Conversation.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

questions = df.question
answers = ["<start> " + answer + " <end>" for answer in df.answer]

# Hyperparameters
VOCAB_SIZE = 1000
MAX_LEN = 20
EMBEDDING_DIM = 64
LSTM_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 100

# Tokenize the text
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(questions + answers)

# Save the tokenizer
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Convert texts to sequences
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Pad the sequences
question_sequences = pad_sequences(question_sequences, maxlen=MAX_LEN, padding='post')
answer_sequences = pad_sequences(answer_sequences, maxlen=MAX_LEN, padding='post')

# Prepare target data shifted by one timestep
decoder_input_data = np.array([seq[:-1] for seq in answer_sequences])
decoder_target_data = np.array([seq[1:] for seq in answer_sequences])

# Encoder
encoder_inputs = Input(shape=(MAX_LEN,))
encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(MAX_LEN - 1,))
decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention layer
attention = Attention()
attention_output = attention([decoder_lstm_outputs, encoder_outputs])

# Concatenate attention output with LSTM output
decoder_combined_context = Concatenate()([attention_output, decoder_lstm_outputs])

# Dense layer for generating the final outputs
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
history = model.fit(
    [question_sequences, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

history_dict = history.history
pd.DataFrame(history_dict).to_csv("data/training_attention.csv", index=False)

# Encoder model for inference
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder model for inference
decoder_state_input_h = Input(shape=(LSTM_UNITS,))
decoder_state_input_c = Input(shape=(LSTM_UNITS,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Attention mechanism with decoder in inference mode
decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)

# Apply attention
attention_output = attention([decoder_lstm_outputs, encoder_outputs])
decoder_combined_context = Concatenate()([attention_output, decoder_lstm_outputs])
decoder_outputs = decoder_dense(decoder_combined_context)

# Define the decoder model
decoder_model = Model(
    [decoder_inputs, encoder_outputs] + decoder_states_inputs,
    [decoder_outputs] + [state_h, state_c, attention_output]
)

# Save the encoder and decoder models
encoder_model.save("models/encoder_attention_model.keras")
decoder_model.save("models/decoder_attention_model.keras")

# Function to generate responses
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')
    encoder_outs, state_h, state_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c, attn = decoder_model.predict([target_seq, encoder_outs, state_h, state_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > MAX_LEN:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        state_h, state_c = h, c

    return decoded_sentence.strip()

# Example usage
input_text = "How are you?"
response = generate_response(input_text)
print("Bot:", response)
