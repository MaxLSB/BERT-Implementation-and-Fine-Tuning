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
    if output == 0:
        return 'Sadness 😟'
    elif output == 1:
        return 'Joy 😁'
    elif output == 2:
        return 'Love 🥰'
    elif output == 3:
        return 'Anger 😱'
    elif output == 4:
        return 'Fear 😱'
    elif output == 5:
        return 'Surprise 😲'
            
def main():
    with st.expander('Details', expanded=True):
        st.title('Bert Emotion Classifier 📝')
        st.caption('By Maxence Lasbordes')
        
        user_input = st.text_input("Enter yout text here:", placeholder = "I am absolutely thrilled about the upcoming trip to Paris!", max_chars=128)
        
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path = 'trained_model/bert_emotion_classifier.pth'
        model = model_load(path).to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if user_input:
            
            input_encoding = tokenizer(user_input, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            logits = model(input_encoding['input_ids'], input_encoding['attention_mask'])
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, 1)
            predicted_emotion = emotion(predicted_class)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write('Predicted Emotion:')
            with col2:
                st.subheader(f'{predicted_emotion}')
            st.write(f'*Probability: {probs[0][predicted_class].item()*100:.2f} %*')
            

    

if __name__ == '__main__':
    main()