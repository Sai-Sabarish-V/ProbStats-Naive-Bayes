# Bayesian Spam Detector

A machine learning project that classifies SMS messages as **Spam** or **Ham (Genuine)** using the Multinomial Naive Bayes algorithm. The project explores Bayesian probability, word-based spam analysis, probability distributions, and model evaluation using the SMS Spam Collection dataset.

## Features

* **Dataset Handling:** Downloads the SMS Spam Collection dataset directly from the UCI Machine Learning Repository.
* **Data Preprocessing:** Checks for missing values, removes duplicate messages, and converts labels into numerical values.
* **Dataset Analysis:** Calculates the prior probabilities of spam and genuine messages and visualizes their distribution.
* **Bayesian Probability Analysis:** Uses Bayes' theorem to investigate the relationship between suspicious words and spam messages.
* **Random Variable Analysis:** Calculates the expected value and variance of a binary random variable representing spam classification.
* **Machine Learning Model:** Trains a Multinomial Naive Bayes classifier using word-frequency features.
* **Model Evaluation:** Evaluates the model using accuracy, a classification report, and a confusion matrix.
* **SMS Prediction Demo:** Predicts whether a new SMS message is spam or genuine and displays the predicted class probabilities.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Regular Expressions
* Jupyter Notebook

## Dataset

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

The dataset contains SMS messages labelled as:

* **Ham (0):** Legitimate messages.
* **Spam (1):** Unwanted or fraudulent messages.

Dataset source: [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The notebook downloads the dataset directly from UCI, checks for missing values, removes duplicate messages, and maps the labels to numerical values.

## Methodology

### 1. Dataset Handling

The dataset is loaded into a Pandas DataFrame with two primary columns:

* `label`: Original message category (`ham` or `spam`).
* `text`: SMS message content.

Duplicate messages are removed, and a numerical label column is created:

* `0` represents Ham.
* `1` represents Spam.

### 2. Exploratory Data Analysis

The project calculates the total number of messages, the number of spam messages, and the number of genuine messages.

The prior probabilities are calculated as:

$$
P(Spam) = \frac{\text{Number of Spam Messages}}{\text{Total Messages}}
$$

$$
P(Ham) = \frac{\text{Number of Ham Messages}}{\text{Total Messages}}
$$

A count plot visualizes the distribution of the two classes.

### 3. Word-Based Bayesian Probability Analysis

The project investigates five suspicious words:

* `free`
* `prize`
* `win`
* `urgent`
* `offer`

Regular expressions identify messages containing each word, ignoring case and matching whole words.

For each word, the notebook calculates the empirical conditional probability:

$$
P(Spam \mid Word) =
\frac{\text{Spam Messages Containing the Word}}
{\text{All Messages Containing the Word}}
$$

It also verifies the result using Bayes' theorem:

$$
P(Spam \mid Word) =
\frac{P(Word \mid Spam)P(Spam)}
{P(Word)}
$$

This analysis demonstrates how the presence of particular words can influence the probability that a message is spam.

### 4. Random Variable Analysis

A binary random variable \(Y\) represents whether a randomly selected message is spam:

$$
Y =
\begin{cases}
1, & \text{if the message is Spam} \\
0, & \text{if the message is Ham}
\end{cases}
$$

Let \(p = P(Spam)\).

The expected value is:

$$
E(Y) = p
$$

The variance is:

$$
Var(Y) = p(1-p)
$$

These calculations connect the dataset's class distribution to the properties of a Bernoulli random variable.

### 5. Data Preparation

The dataset is divided into training and testing sets using an 80:20 split.

The implementation uses:

* `test_size=0.2`
* `random_state=42`
* Stratified sampling to preserve the class proportions.

`CountVectorizer` converts SMS messages into numerical word-frequency features. English stop words are excluded.

The vectorizer is fitted on the training messages and then used to transform the testing messages.

### 6. Model Building

The project uses the **Multinomial Naive Bayes** classifier from Scikit-learn.

The model is trained on the vectorized SMS messages and their corresponding numerical labels.

Naive Bayes applies Bayes' theorem while assuming conditional independence between features given the class. Multinomial Naive Bayes is commonly used for text classification tasks involving word counts.

### 7. Model Evaluation

The trained model predicts the classes of messages in the testing set.

The notebook evaluates the model using:

* **Accuracy:** Proportion of correctly classified messages.
* **Precision:** Proportion of messages predicted as a class that actually belong to that class.
* **Recall:** Proportion of messages belonging to a class that are correctly identified.
* **F1-score:** Harmonic mean of precision and recall.
* **Confusion Matrix:** Displays true negatives, false positives, false negatives, and true positives.

The confusion matrix also helps identify two important types of errors:

* **False Positive:** A genuine message is incorrectly classified as spam.
* **False Negative:** A spam message is incorrectly classified as genuine.

The notebook prints the classification report and visualizes the confusion matrix.

### 8. SMS Prediction Demo

The `predict_sms()` function accepts a message as input and returns:

* The input message.
* The predicted class: Spam or Ham.
* The predicted spam probability.
* The predicted genuine-message probability.

Example:

```python
predict_sms("Congratulations! You have won a free prize. Claim now.")
predict_sms("Hey, how are you?")
```

The function transforms each input message using the previously fitted vectorizer and passes the resulting features to the trained model.

## Installation and Setup

### Prerequisites

* Python 3.9 or later
* Jupyter Notebook or JupyterLab
* Internet access to download the dataset

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your repository URL and folder name.

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `bayesian_spam_detector.ipynb` and run the cells sequentially.

The notebook downloads the dataset automatically, so a separate dataset download is not required.

## Project Structure

```text
Bayesian-Spam-Detector/
│
├── bayesian_spam_detector.ipynb
└── README.md
```

## Results

The notebook generates the following outputs:

* Spam and Ham message counts and prior probabilities.
* A visualization of the class distribution.
* Word-based conditional probability calculations.
* Expected value and variance of the spam indicator random variable.
* A trained Multinomial Naive Bayes classifier.
* Accuracy and a classification report.
* A confusion matrix.
* Example predictions for new SMS messages.

**Note:** The numerical evaluation results depend on executing the notebook. No specific accuracy or performance score is claimed here because the notebook's recorded outputs have not been provided.

## Learning Outcomes

This project demonstrates:

1. Applying probability theory to a real-world classification problem.
2. Using Bayes' theorem to calculate conditional probabilities.
3. Understanding Bernoulli random variables, expectation, and variance.
4. Converting textual data into numerical features using `CountVectorizer`.
5. Training a Multinomial Naive Bayes classifier.
6. Evaluating classification models using standard metrics.
7. Building a basic SMS spam prediction pipeline.

## Limitations

* The model uses word-frequency features and does not understand the semantic meaning of messages.
* Unseen vocabulary and changes in spam patterns may affect predictions.
* The word-based probability analysis examines individual suspicious words rather than combinations of words.
* The independence assumption of Naive Bayes is a simplification of natural language.
* Predicted probabilities are model estimates and should not automatically be interpreted as perfectly calibrated real-world probabilities.

## License

Specify a license for this project before distributing or reusing its code. If no license is provided, the repository's reuse permissions remain unspecified.
