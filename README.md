# Hate Speech Analysis

This project focuses on **hate speech detection** using a variety of machine learning models. It addresses the problem of distinguishing between hate speech and free speech, particularly in online contexts, where discriminatory or violent content is more prevalent.

## Overview

The project uses the **"tweets hate speech detection"** dataset, sourced from Hugging Face, to classify tweets as containing hate speech (racist or sexist content) or not. Out of a dataset of **31,962 tweets**, we selected a subset of **10,000** for training and testing. The final goal is to develop an efficient model that can assist in moderating online platforms by automatically flagging harmful content.

## Features

- **Dataset**: The dataset is preprocessed through steps such as converting text to lowercase, removing hyperlinks, usernames, hashtags, punctuation, stopwords, and lemmatization to enhance the detection accuracy.
- **Feature Extraction**: 
  - **CBoW (Continuous Bag of Words)**: A neural network-based method that reconstructs word context.
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Measures word importance in the dataset.
  
## Classifier Models

We implemented and tested a range of classifier models:

1. **Multinomial Naive Bayes**
2. **Ridge Classifier**
3. **SVC (kernel rbf)**
4. **Linear SVC**
5. **Gradient Boosting**
6. **MLP Classifier**
7. **XGBoost**
8. **Pre-trained BERT Models** from Hugging Face
9. **Deep Neural Network** (DNN) with two hidden layers

Each model was fine-tuned using techniques like **n-grams**, and **GridSearchCV** for hyperparameter tuning.

## Results

The top-performing models in terms of **macro F1 score**:
- **Linear SVC** and **MLP Classifier**: ≈ 88%
- **Ridge Classifier**: ≈ 86%
- **XGBoost**: ≈ 83%
- **DNN**: ≈ 83%

## Observations

- **N-grams** boosted model performance by about 6%.
- Including hashtags improved accuracy by 5%.
- Pre-trained models had lower effectiveness due to dataset-specific subjectivity in labeling hate speech.

## Future Work

While the models show promising results, further refinement is needed, especially in reducing **false positives** and improving **false negatives**, to ensure hate speech is flagged correctly without infringing on free speech.
