#!/bin/bash

mkdir -p trained_model

url="https://www.dropbox.com/scl/fi/yhkhxtxq9qggfcci7j535/bert_emotion_classifier.pth?rlkey=pf5dnuzw6bjhtet4fx0te1jep&st=h9e3frps&dl=1"

filename="bert_emotion_classifier.pth"

curl -L -o "trained_model/${filename}" "${url}"

echo "Downloaded the model weights to /trained_model/"
