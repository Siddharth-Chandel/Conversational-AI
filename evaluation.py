import pandas as pd
import seaborn as sns
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
import evaluate
import numpy as np
from keras.models import load_model
import pickle

MAX_LEN = 20

# Load the metrics
rouge_metric = evaluate.load('rouge')

# Load the tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load encoder and decoder models
encoder_model = load_model("models/encoder_model.keras")
decoder_model = load_model("models/decoder_model.keras")
encoder_attention_model = load_model("models/encoder_attention_model.keras")
decoder_attention_model = load_model("models/decoder_attention_model.keras")

# Function to generate responses
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

# Function to generate responses and capture attention weights
def generate_attention_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding='post')
    encoder_outs, state_h, state_c = encoder_attention_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']

    decoded_sentence = ''
    attention_weights = []  # To store attention weights for each word

    stop_condition = False
    while not stop_condition:
        output_tokens, h, c, attn = decoder_attention_model.predict([target_seq, encoder_outs, state_h, state_c])
        attention_weights.append(attn[0, -1, :])  # Store attention weights for each timestep

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) > MAX_LEN:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        state_h, state_c = h, c

    return decoded_sentence.strip(), attention_weights

# Function to plot the attention weights
def plot_attention(input_text, decoded_sentence, attention_weights):
    input_seq = input_text.split()
    output_seq = decoded_sentence.split()

    # Convert attention weights to a numpy array
    attention_weights = np.array(attention_weights)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights[:len(output_seq), :len(input_seq)], 
                xticklabels=input_seq, yticklabels=output_seq, 
                cmap="YlGnBu", cbar=True, annot=True)
    plt.xlabel("Input Sequence")
    plt.ylabel("Output Sequence")
    plt.title(f"Attention Weights\nUser Input: {input_text}\nResponse: {decoded_sentence}")
    plt.show()

# Evaluate metrics with attention
def evaluate_metrics(test_df):
    rouge_scores = []
    rouge_scores_attention = []

    for _, row in test_df.iterrows():
        predicted_response = generate_response(row['question'])
        predicted_response_attention,  _ = generate_attention_response(row['question'])
        reference_response = row['answer']
        rouge_scores.append(rouge_metric.compute(predictions=[predicted_response], references=[reference_response])['rougeL'])
        rouge_scores_attention.append(rouge_metric.compute(predictions=[predicted_response_attention], references=[reference_response])['rougeL'])

    return {
        'rouge': np.mean(rouge_scores),
        'rouge-attention': np.mean(rouge_scores_attention)
    }

test_df = pd.read_csv("data/test_data.csv")
metrics = evaluate_metrics(test_df)
print(metrics)

# Plotting training loss and metrics
def plot_training(history, metrics:dict):
    epochs = range(1, len(history[0]['loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 1)
    plt.plot(epochs, history[0]['loss'], label="LSTM")
    plt.plot(epochs, history[1]['loss'], label="LSTM + Attention")
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot metrics
    ax2 = plt.subplot(1, 2, 2)
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.title('Metrics Over Epochs')
    plt.xlabel('Metrics')
    plt.ylabel('Avg. Scores')
    # Remove axes splines
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    ax2.xaxis.set_tick_params(pad = 5)
    ax2.yaxis.set_tick_params(pad = 10)

    plt.show()


# Plot the training history and metrics
training_history = pd.read_csv("data/training.csv")
training_history_attention = pd.read_csv("data/training_attention.csv")
plot_training([training_history, training_history_attention], metrics)

text = 'how are you doing today?'
response, attn = generate_attention_response(text)
plot_attention(text, response, attn)