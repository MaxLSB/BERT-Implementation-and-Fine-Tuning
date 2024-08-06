#!/bin/bash

mkdir -p trained_model

url="https://www.dropbox.com/scl/fi/nwrim7ssxxzqukuv02y3c/bert_emotion_classifier.pth?rlkey=8s9g1rtryz3g4mvdni953bksw&st=j7swq1jq&dl=0"

filename="bert_emotion_classifier.pth"

curl -L -o "trained_model/${filename}" "${url}"

echo "Downloaded the model weights to /trained_model/"
