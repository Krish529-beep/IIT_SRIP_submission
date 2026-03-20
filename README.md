# Topic Classification Project

This project builds a machine learning model to classify text into predefined topics. The model takes text as input and predicts the most relevant topic.

---

## Overview

- Uses classical machine learning techniques  
- Handles large dataset using chunk-based loading  
- Uses TF-IDF for feature extraction  
- Uses SGDClassifier for efficient training  

---

## Project Structure

| Folder/File         | Description |
|-------------------|------------|
| src/              | Source code (training and inference) |
| final_models/     | Saved model and vectorizer |
| report.pdf        | Project report |
| requirements.txt  | Dependencies |
| README.md         | Documentation |

---

## System Specs

- Python 3.13 
- 16 GB RAM 
- Windows 11 
- rx 6700xt 12GB

---

## Setup Instructions

### 1. Clone Repository

git clone <your-repo-link>  
cd topic-classifier  

---

### 2. Create Virtual Environment

python -m venv venv  
venv\Scripts\activate  

---

### 3. Install Dependencies

pip install -r requirements.txt  

---

## Training the Model

Run the following command:

python src/train.py  

### Steps performed:
- Load dataset in chunks  
- Clean and preprocess text  
- Convert text to TF-IDF features  
- Train model and evaluate performance  

---

## Inference

Run:

python src/inference.py  

Enter any text and the model will predict the topic.

---

## Model Details

| Component        | Description |
|-----------------|------------|
| Feature Method  | TF-IDF (Unigrams + Bigrams) |
| Model           | SGDClassifier |
| Training Data   | ~500,000 samples |
| Accuracy        | ~0.84 |

---

## Results

| Metric            | Value |
|------------------|------|
| Accuracy         | ~0.84 |
| Weighted F1 Score| ~0.83 |
| Macro F1 Score   | ~0.66 |

---

## Sample Output

<img width="1041" height="207" alt="image" src="https://github.com/user-attachments/assets/0d77315e-3324-4bb7-976d-2861b733022b" />


---

## Model Files

The trained model files are included in the repository:

| File            | Description |
|----------------|------------|
| model.pkl       | Trained classification model |
| vectorizer.pkl  | TF-IDF vectorizer |

Place these files inside the final_models/ folder before running inference.

---

## Notes

- Full dataset was not used due to memory constraints  
- Chunk-based loading was used to handle large data  

---

## Author

Krish Brahmbhatt
