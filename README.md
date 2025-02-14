# gaussian-vs-multinomial-nb
# Naïve Bayes Classifier: Gaussian vs. Multinomial

## Overview
This project implements and compares **Gaussian Naïve Bayes (GNB)** and **Multinomial Naïve Bayes (MNB)** classifiers on the **Iris dataset**. Since the Iris dataset contains continuous numerical features, we expect **GaussianNB** to perform better than **MultinomialNB**.

## Features
- Loads the **Iris dataset** from `sklearn.datasets`.
- Splits the dataset into **training (80%)** and **testing (20%)** sets.
- Implements **GaussianNB** (for continuous data) and **MultinomialNB** (for discrete data).
- Evaluates performance using:
  - **Accuracy**
  - **Confusion Matrix**
  - **Classification Report**
- Provides **PCA-based visualization** for decision boundaries.

## Installation
### Prerequisites
Ensure you have Python installed along with the necessary dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage
### Running the Classifier
Clone this repository and execute the script:
```bash
git clone https://github.com/yourusername/naive-bayes-iris.git
cd naive-bayes-iris
python compare_naive_bayes.py
```

### Expected Output
- **Accuracy Scores for Both Models**
- **Confusion Matrices for Each Classifier**
- **Classification Reports with Precision, Recall, and F1-Score**
- **PCA Visualization of Decision Boundaries**

## Implementation Details
### Gaussian Naïve Bayes (GNB)
- Computes **mean** and **standard deviation** for each feature per class.
- Uses the **Gaussian probability density function** for likelihood calculation.
- Predicts class labels using **Bayes' Theorem**.

### Multinomial Naïve Bayes (MNB)
- Assumes **features represent discrete frequency counts** (not suitable for Iris dataset).
- Uses **categorical probability distribution**.
- Trains and predicts using probability mass functions.

## Example Output
```
Gaussian Naïve Bayes Model Evaluation:
Accuracy: 96.67%
Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
Classification Report:
  Precision  Recall  F1-Score
Setosa         1.00    1.00     1.00
Versicolor     0.90    0.90     0.90
Virginica      1.00    1.00     1.00

Multinomial Naïve Bayes Model Evaluation:
Accuracy: 73.33%
Confusion Matrix:
[[10  0  0]
 [ 0  6  4]
 [ 0  4  6]]
Classification Report:
  Precision  Recall  F1-Score
Setosa         1.00    1.00     1.00
Versicolor     0.60    0.60     0.60
Virginica      0.67    0.67     0.67
```

## Visualization
The script generates:
- **Accuracy Bar Chart**: Comparison of both models.
- **Confusion Matrices**: Heatmaps for classification performance.
- **PCA-based Decision Boundary Plot**.

## Conclusion
- **GaussianNB performs better** on the Iris dataset due to continuous feature values.
- **MultinomialNB struggles**, as it assumes categorical or count-based data.
- The difference is evident in the accuracy and confusion matrices.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions
Contributions are welcome! Feel free to open issues and submit pull requests.

## Author
[D. Anu Kumari](https://github.com/942004)

