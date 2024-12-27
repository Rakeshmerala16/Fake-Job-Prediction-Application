Fake Job Prediction Application

Overview

This project implements a machine learning model to predict whether a job posting is fake or real based on patterns and features extracted from the data. The model leverages text and numerical features to analyze job postings and identify fraudulent listings, providing a robust solution for job-seekers and platforms.

Features

Data Cleaning: Handles missing values, removes irrelevant data, and preprocesses text fields.

Feature Extraction:

Text features are processed using TF-IDF to capture word importance.

Numerical features such as telecommuting and has_company_logo are used for context.

Hybrid Model: Combines multiple base algorithms (Logistic Regression, Random Forest, Gradient Boosting) with a Stacking Classifier to enhance prediction accuracy.

Evaluation: Model performance is measured using accuracy, precision, recall, and F1-score.

Prediction Pipeline: Accepts new job postings, processes their features, and predicts their authenticity.

Dataset

The model uses a dataset with labeled job postings containing:

Text Fields: Job title, description, and other textual data.

Numerical Fields: Indicators like telecommuting, presence of a company logo, and screening questions.

Example Features:

Fake Jobs:

Keywords: "Earn," "Easy Money," "Work from Home."

Lack of company details or logos.

Unrealistic benefits or vague descriptions.

Real Jobs:

Specific job titles and descriptions.

Company profiles with logos and screening questions.

Reasonable benefits and qualifications.

How It Works

1. Data Preparation

Clean the dataset to handle missing values and irrelevant entries.

Combine text-based fields into a single feature for better context analysis.

2. Feature Extraction

Text Features: Convert text to numerical vectors using TF-IDF.

Numerical Features: Use non-text columns like telecommuting and company logo presence.

3. Hybrid Model Construction

Base models:

Logistic Regression: Captures linear relationships.

Random Forest: Identifies complex patterns using ensembles of decision trees.

Gradient Boosting: Focuses on misclassified samples to improve predictions.

A meta-classifier (Logistic Regression) combines base model predictions for a final decision.

4. Training and Evaluation

Train the model using labeled data to learn patterns.

Test on unseen data to evaluate metrics like accuracy, precision, recall, and F1-score.

5. Prediction Pipeline

Input Processing: Text fields are vectorized using TF-IDF, and numerical features are appended.

Prediction: Base models make predictions, and the meta-classifier aggregates them to provide the final label.

Example Usage

Fake Job Input:

{
  "title": "Work from Home - Easy Money",
  "description": "Earn $500 per day with no experience.",
  "telecommuting": true,
  "has_company_logo": false,
  "has_screening_questions": false
}

Prediction: Fake

Real Job Input:

{
  "title": "Full-Stack Developer",
  "description": "Develop and maintain applications. 3+ years experience required.",
  "telecommuting": false,
  "has_company_logo": true,
  "has_screening_questions": true
}

Prediction: Real
