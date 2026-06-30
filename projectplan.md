A Respiratory Disease prediction, analysis and forecasting cross-platform app with widget and notification facility.

\\

A multimodal respiratory disease risk prediction and analysis system using chest X-rays, breathing audio, EHR records, and patient information, capable of classifying selected respiratory conditions and generating future risk alerts.



input: 4 types-

|Chest X-Rays|
|-|
|Breathing Audio Samples|
|EHR-Excel Records|
|Synthetic Patient Information|



Model Evaluation

Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC.



1. EHR:

import pandas as pd

df = pd.read\_csv("dataset.data/synthetic\_patient\_records/synthetic\_respiratory\_patient\_records.csv")



2\. X-rays:

2 megasets



3\. breathing audios:

WAV-machine readable files



4\. Synthetic patient info:

UI inputs







1\. Chest X-rays

CNN image classification.

Classes from Kaggle/current datasets: Normal, Pneumonia types, COVID-19, TB, etc.



2\. Breathing Audio Samples

Audio classification.

Needs quality inspection + metadata/labels, but the raw input source exists.



3\. EHR / Excel Records

Structured tabular ML.

Synthetic 1,500-row dataset already created for development.



4\. Patient Information

Direct UI questionnaire.

Bio-data, symptoms, lifestyle, AQI/location, history, vitals, and user context.





