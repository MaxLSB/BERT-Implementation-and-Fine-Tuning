<div align="center">
  <img src="assets/banner.png" alt="Example" width="1000"/>
</div>

# Introduction

I re-implemented BERT from scratch using the paper as a reference. I trained a custom tokenizer, cleaned and prepared a movie dialogue dataset, and built the pre-processing of the data for the NSP and MLM tasks. I also developed the input-embedding functions, the BERT model (transformer encoder + NSP + MLM), and the trainer, using PyTorch. Unfortunately, I wasn't able to train the model due to computational constraints. However, I successfully completed the model architecture and preprocessing pipeline, and validated the implementation with smaller-scale tests to ensure functionality.

To complete the process, I fine-tuned the pre-trained BERT model available on Hugging Face for emotion classification on a dataset of 30,000 posts, achieving 95% accuracy. I deployed this model using Streamlit.

**You can access and test the model [here](https://bert-emotion-classification.streamlit.app/)**

_(Please also check: notebook.ipynb)_

# Installation ✨

Use a dedicated environnement to install the librairies.

Clone repo :
```
git clone https://github.com/MaxLSB/BERT_LLM_From_Scratch.git
```
Install the requirements for the app:
```
pip install -r requirements.txt
```

# Streamlit server

<div align="center">
  <img src="assets/streamlit-app.png" alt="Example" width="800" />
</div>

You can access and test the model here:
```
https://bert-emotion-classification.streamlit.app/
```

