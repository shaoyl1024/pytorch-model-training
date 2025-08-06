# Display Advertising CTR Prediction

## Project Overview

This project focuses on predicting the Click-Through Rate (CTR) for display advertisements using the Criteo Display
Advertising Challenge Dataset. The goal is to build effective predictive models that can estimate the probability of a
user clicking on a displayed ad, which is crucial for optimizing ad performance, user experience, and advertising
revenue.

## Dataset Information

### Data Source

The project utilizes the Criteo Display Advertising Challenge Dataset, which contains anonymized click-through data from
Criteo's advertising platform. This dataset is widely used in the machine learning community for benchmarking CTR
prediction models.

### Dataset Characteristics

* **Time Period:** The data covers 1 week of advertising interactions.

* **Format:** Structured tabular data with labeled examples (supervised learning setup).
* **Anonymization:** All personally identifiable information (PII) and sensitive data have been removed. Categorical
  features are hashed to ensure privacy, meaning original values cannot be reconstructed while retaining their
  predictive value.
* **Size:** The original full dataset includes approximately 45 million training examples, making it a large-scale
  dataset suitable for testing scalable machine learning approaches.

### Data Fields

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

> Note: The anonymization process ensures that all features are privacy-preserving, with no ability to reverse-engineer
> personal or sensitive information. This allows for safe experimentation while maintaining the dataset's utility for
> predictive modeling.

## Key Objectives

* Develop machine learning models to predict the likelihood of ad clicks (CTR).
* Evaluate model performance using industry-standard metrics for classification tasks.
* Explore feature engineering and preprocessing techniques suitable for large-scale tabular data with mixed numerical
  and categorical features.

## Usage

Details on data preprocessing, model training, and evaluation can be found in the project's codebase. The dataset can be
obtained from the official Criteo Display Advertising Challenge sources (please refer to the original dataset
documentation for access).

