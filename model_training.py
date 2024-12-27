import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from scipy.sparse import hstack

# Load the dataset
data = pd.read_csv('data/fake_job_postings.csv')

# Initial data inspection
print("Initial dataset shape:", data.shape)
print("Columns in the dataset:", data.columns)

# Drop irrelevant columns
irrelevant_columns = ['job_id', 'location', 'department', 'salary_range']
data.drop(columns=irrelevant_columns, inplace=True, errors='ignore')

# Remove rows with missing target values
data = data.dropna(subset=['fraudulent'])

# Fill missing values in text columns with empty strings
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    data[col] = data[col].fillna('')

# Fill missing values in numerical columns with 0
numerical_columns = ['telecommuting', 'has_company_logo', 'has_questions']
for col in numerical_columns:
    data[col] = data[col].fillna(0)

# Combine text columns into a single feature
data['text_features'] = data['title'] + ' ' + \
                        data['company_profile'] + ' ' + \
                        data['description'] + ' ' + \
                        data['requirements'] + ' ' + \
                        data['benefits']

# Select relevant features and target
X = data[['text_features', 'telecommuting', 'has_company_logo', 'has_questions']]
y = data['fraudulent']

# Ensure class balance
print("Class distribution:")
print(y.value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF vectorizer for text features
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_text = tfidf.fit_transform(X_train['text_features'])
X_test_text = tfidf.transform(X_test['text_features'])

# Combine text features with numerical features
X_train_combined = hstack([X_train_text, X_train[['telecommuting', 'has_company_logo', 'has_questions']].values])
X_test_combined = hstack([X_test_text, X_test[['telecommuting', 'has_company_logo', 'has_questions']].values])

# Define base models
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Hybrid model: Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
    final_estimator=LogisticRegression(),
    cv=5
)

# Train the hybrid model
print("Training the hybrid model...")
stacking_model.fit(X_train_combined, y_train)

# Evaluate the model
y_pred = stacking_model.predict(X_test_combined)
print("Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('saved_models/hybrid_model.pkl', 'wb') as model_file:
    pickle.dump({'model': stacking_model, 'vectorizer': tfidf}, model_file)

print("Model and vectorizer saved successfully!")
