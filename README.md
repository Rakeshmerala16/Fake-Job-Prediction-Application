******Fake Job Prediction Application******

****Overview****

    This project uses a machine learning model to predict whether a job posting is fake or real based on various input fields. The system combines text analysis and numerical features to provide a reliable decision-making process, benefiting both job-seekers and job platforms.

****Features****

**Input Fields:**

**Job Title:** Captures keywords indicative of legitimacy or fraud.

**Company Profile:** Assesses the level of detail in the company's description.

**Job Description:** Evaluates responsibilities and language used.

**Requirements:** Checks for realistic qualifications.

**Benefits:** Identifies plausible or overly attractive offers.

**Telecommuting:** Indicates whether remote work is available.

**Company Logo:** Considers the presence of a company logo.

**Has Screening Questions:** Adds credibility to the listing.

**Hybrid Model:**

    Combines predictions from Logistic Regression, Random Forest, and Gradient Boosting.

    Utilizes a Stacking Classifier for enhanced decision-making.

**Output:**

  Predicts whether a job is Fake or Real.

  Provides probabilities for each class.

****How It Works****

**1. Data Preparation**

Cleans the dataset to handle missing values.

Combines text-based fields into a single feature for analysis.

**2. Feature Extraction**

Text Features: Converts text into numerical vectors using TF-IDF.

Numerical Features: Appends binary indicators like telecommuting, logo presence, and screening questions.

**3. Model Training**

Trains a hybrid model using labeled data to learn patterns of fake and real job postings.

**4. Prediction Pipeline**

Processes new job postings with the same TF-IDF and numerical feature pipeline.

Passes the features through the hybrid model to make predictions.

**Example Usage**

**Input:**

**Fake Job Input:**

{
  "job_title": "Work from Home - Earn Money Fast",
  "company_profile": "",
  "job_description": "Earn $1000 per week with no experience required.",
  "requirements": "",
  "benefits": "Unlimited earnings, no office hours.",
  "telecommuting": true,
  "company_logo": false,
  "has_questions": false
}

****Prediction: Fake****

**Real Job Input:**

{
  "job_title": "Senior Data Scientist",
  "company_profile": "A reputable data analytics company.",
  "job_description": "Develop predictive models, work with large datasets.",
  "requirements": "3+ years of experience in data science, Python, SQL.",
  "benefits": "Competitive salary, health insurance, remote work.",
  "telecommuting": true,
  "company_logo": true,
  "has_questions": true
}

****Prediction: Real****
