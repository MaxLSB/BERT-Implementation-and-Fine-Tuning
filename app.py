import streamlit as st
import torch
from transformers import AutoModel, BertTokenizer
from models.classifier import EmotionClassifier
import torch.nn.functional as F
import requests
from io import BytesIO

# Load model and tokenizer once and reuse
@st.cache_resource  # Cache the model and tokenizer to avoid reloading
def load_model_and_tokenizer():
    url = "https://www.dropbox.com/scl/fi/yhkhxtxq9qggfcci7j535/bert_emotion_classifier.pth?rlkey=pf5dnuzw6bjhtet4fx0te1jep&st=h9e3frps&dl=1"
    
    # Download the model weights
    response = requests.get(url)
    response.raise_for_status()
    model_weights = BytesIO(response.content)

    # Initialize the model and tokenizer
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = EmotionClassifier(bert, 6)
    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model.eval()
    return model, tokenizer

def emotion(output):
    emotions = {
        0: 'Sadness 😟',
        1: 'Joy 😁',
        2: 'Love 🥰',
        3: 'Anger 😡',
        4: 'Fear 😱',
        5: 'Surprise 😲'
    }
    return emotions.get(output.item(), "Unknown")

def main():
    st.set_page_config(page_title="Emotion Classifier", page_icon=":smiley:", layout="centered")
    
    model, tokenizer = load_model_and_tokenizer()
    
    st.image('assets/streamlit-banner.png', use_column_width=True)
    st.header('BERT Emotion Classifier 📝')
    
    user_input = st.text_input("Enter your text here (english only):", placeholder="I am absolutely thrilled about the upcoming trip to Paris!", max_chars=128)
    
    if user_input:
        with st.spinner('Analyzing...'):
            input_encoding = tokenizer(user_input, truncation=True, max_length=128, return_tensors='pt')
            input_ids = input_encoding['input_ids']
            attention_mask = input_encoding['attention_mask']
            
            # use the appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
            
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, 1)
            predicted_emotion = emotion(predicted_class)
            
            # Display the result   
            st.success(f"Predicted emotion: {predicted_emotion}")
            # Display the probability
            st.info(f"Confidence: {probs[0][predicted_class].item() * 100:.2f} %")
            
    st.write("---")
    st.write("### About")
    st.write("This app uses a fine-tuned BERT model to classify emotions based on text input. The model is trained to recognize 6 different emotions: sadness, joy, love, anger, fear, and surprise.")

    st.caption('By Maxence Lasbordes')
    
if __name__ == '__main__':
    main()
