import streamlit as st
import torch
from transformers import AutoModel, BertTokenizer
from models.classifier import EmotionClassifier
import torch.nn.functional as F

def model_load(path):
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = EmotionClassifier(bert, 6)
    model.load_state_dict(torch.load(path))
    return model

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
    
    with st.expander('Model', expanded=True):
        
        
        st.title('BERT Emotion Classifier 📝')
        st.caption('By Maxence Lasbordes')
        
        user_input = st.text_input("Enter your text here:", placeholder="I am absolutely thrilled about the upcoming trip to Paris!", max_chars=128)
        
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path = 'trained_model/bert_emotion_classifier.pth'
        
        try:
            model = model_load(path).to(device)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
        # If the user has entered a text
        if user_input:
            with st.spinner('Analyzing...'):
                input_encoding = tokenizer(user_input, truncation=True, max_length=128, return_tensors='pt').to(device)
                
                logits = model(input_encoding['input_ids'], input_encoding['attention_mask'])
                probs = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probs, 1)
                predicted_emotion = emotion(predicted_class)
                
                # Display the results   
                st.markdown(
                f'''
                <div style="background-color: #303030; text-align: center; padding: 10px; border-radius: 5px;">
                    <h2 style="font-weight: bold; margin: 0; color: #e0e0e0;">{predicted_emotion}</h2>
                </div>
                ''',
                unsafe_allow_html=True
                )
                
                # Display the probability
                st.markdown(
                f'''
                <div style="padding-top: 20px;">
                    <p style="font-style: italic;">Probability: {probs[0][predicted_class].item() * 100:.2f} %</p>
                </div>
                ''',
                unsafe_allow_html=True
                )
                
        st.write("---")
        st.write("### About")
        st.write("This app uses a pre-trained BERT model to classify emotions based on text input. The model is trained to recognize various emotions such as sadness, joy, love, anger, fear, and surprise.")

if __name__ == '__main__':
    main()
