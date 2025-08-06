# Display Advertising CTR Prediction

# Project Overview

This project aims to predict the Click-Through Rate (CTR) of display advertisements using the Criteo Display Advertising
Challenge Dataset. The core objective is to develop robust predictive models that estimate the probability of a user
clicking on a displayed ad. Accurate CTR prediction is critical for optimizing ad delivery, enhancing user experience,
and maximizing advertising revenue for platforms and advertisers.

# Project Structure

```text
ctr_prediction/
├── notebooks/           # Jupyter Notebooks for interactive data analysis and experimentation
├── data/                # Project data (strict separation of raw and processed data)
├── src/                 # Reusable source code and utility functions
├── figures/             # Generated visualizations and charts
├── models/              # Trained model artifacts (serialized models)
├── requirements.txt     # Explicit list of project dependencies with version pins
├── .gitignore           # Files/folders to exclude from version control (large data, env files)
└── README.md            # Project documentation (this file)
```

# Dataset Information

## Source

The dataset is derived from the [Criteo Display Advertising Challenge](https://ailab.criteo.com/ressources/), a public
benchmark dataset widely used for advancing research on CTR prediction and large-scale machine learning.

## Key Characteristics

* **Time Coverage:** The dataset includes 7 days of ad impression data from Criteo's advertising platform.
* **Scale:** The full training set contains approximately 45 million examples, making it suitable for testing
  scalability of machine learning pipelines.
* **Anonymization:** All features are anonymized to protect user privacy. Categorical features are hashed to prevent
  reconstruction of sensitive information (e.g., user IDs, ad IDs), while retaining predictive value.
  Supervised Learning Setup: Each example is labeled with a binary indicator of whether a click occurred, enabling
  supervised classification.

## Data Fields

The dataset comprises the following anonymized features:

* **Label:** A binary indicator representing whether a click occurred:
    * 1 = Click (the user clicked on the ad)
    * 0 = No-click (the user did not click on the ad)
* **Numerical Features (13 fields):**
    * I1 to I13: Continuous or integer-valued features (pre-normalized to facilitate model training). These may
      represent various user, ad, or context-related metrics (e.g., frequency of ad impressions, user engagement
      metrics).
* **Categorical Features (26 fields):**
    * C1 to C26: Hashed categorical features with large cardinalities. These could correspond to attributes like user
      IDs, ad IDs, device types, or contextual categories, but their original meanings are anonymized via hashing.

# Key Objectives

1. **Develop High-Performance CTR Models:** Build and compare machine learning models (e.g., logistic regression,
   gradient boosting, deep learning) to achieve accurate CTR prediction.
2. **Address Large-Scale Data Challenges:** Implement efficient preprocessing and modeling pipelines to handle
   high-dimensional, large-scale tabular data with mixed feature types (numerical + categorical).
3. **Feature Engineering for CTR Prediction:** Explore feature transformation techniques tailored to CTR prediction (
   e.g., target encoding for categorical features, interaction features) to enhance model performance.
4. **Rigorous Evaluation:** Assess models using industry-standard metrics (AUC-ROC, log loss, precision-recall) and
   analyze performance across subpopulations to ensure robustness.

# Usage Guidelines

## Environment Setup

1. Clone the repository and navigate to the project root.
2. Install dependencies using:

```text
pip install -r requirements.txt
```

## Data Preparation

* The raw dataset (train.txt) should be placed in data/raw/.
* Run notebooks/01_data_loading.ipynb to validate and load data, followed by 02_exploratory_analysis.ipynb for EDA.

## Model Development

* Follow the notebooks in sequence (03 → 04 → 05) to execute feature engineering, model training, and evaluation.
* Reusable logic in src/ can be imported into notebooks for consistent implementation.

# Notes

* **Data Privacy:** The dataset’s anonymization ensures compliance with privacy standards (e.g., GDPR). No sensitive or
  personally identifiable information (PII) is present.
* **Reproducibility:** All code and workflows are designed for reproducibility. Dependencies are pinned in
  requirements.txt, and raw data remains immutable to ensure consistent results.
* **Scalability:** The project structure supports scaling to larger datasets by separating data loading/processing logic
  from analysis, enabling efficient handling of big data.

