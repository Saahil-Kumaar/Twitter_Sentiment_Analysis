# Twitter Sentiment Analysis Using Machine Learning

This project performs **Sentiment Analysis** on tweets using the **Sentiment140 dataset**. It uses natural language processing (NLP) techniques and a **Logistic Regression** model to classify tweets as **positive** or **negative**.

---

## 📂 Dataset

- **Source**: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size**: 1.6 million tweets
- **Fields**:
  - `target`: 0 = negative, 4 = positive (converted to 1)
  - `ids`: Tweet ID
  - `date`: Date of the tweet
  - `flag`: Query (not used)
  - `user`: Username
  - `text`: Actual tweet text

---

## 🧰 Libraries Used

- `numpy`
- `pandas`
- `re`
- `nltk` (for stopwords and stemming)
- `sklearn` (for feature extraction, model training & evaluation)
- `pickle` (for model saving/loading)

---

## 🛠️ Installation & Setup

```bash
# Install dependencies
pip install kaggle numpy pandas scikit-learn nltk

# Setup Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d kazanova/sentiment140

# Extract the ZIP file
unzip sentiment140.zip
````

---

## 🔍 Workflow

1. **Load and Prepare Dataset**
   Load the CSV file and rename columns for clarity.

2. **Data Preprocessing**

   * Remove punctuation and non-alphabet characters.
   * Convert text to lowercase.
   * Tokenize and remove stopwords.
   * Apply stemming using `PorterStemmer`.

3. **Feature Extraction**
   Use **TF-IDF Vectorizer** to convert text data into numerical format.

4. **Train/Test Split**
   Split dataset into 80% training and 20% testing.

5. **Model Training**
   Train a **Logistic Regression** model using the processed data.

6. **Evaluation**
   Compute **accuracy score** for both training and test data.

7. **Model Serialization**
   Save the trained model using `pickle`.

8. **Prediction**
   Load the saved model and make predictions on new tweets.

---

## ✅ Results

* **Training Accuracy**: \~79.871953125%
* **Testing Accuracy**: \~77.668125%

> These results may vary slightly based on the training-test split and environment.

---

## 💡 Example Prediction

```python
X_new = X_test[200]
prediction = loaded_model.predict(X_new)

if prediction[0] == 0:
    print("Negative Tweet")
else:
    print("Positive Tweet")
```

---

## 📁 File Structure

```
├── sentiment140.zip
├── training.1600000.processed.noemoticon.csv
├── kaggle.json
├── trained_model.sav
└── sentiment_analysis.py  # (optional: if code saved as script)
```

---

## 🚀 Future Improvements

* Use more advanced models like SVM, XGBoost, or deep learning (LSTM, BERT).
* Incorporate emojis, hashtags, and mentions into sentiment context.
* Create a web app using Streamlit or Flask.

---

## 📜 License

This project is for educational purposes and uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) under the respective license.

---

## 🙌 Acknowledgements

* [Kaggle](https://www.kaggle.com/)
* [NLTK](https://www.nltk.org/)
* [Scikit-learn](https://scikit-learn.org/)

```

---

Let me know if you'd like a version with badges, Streamlit app support, or GitHub Actions integration!
```
