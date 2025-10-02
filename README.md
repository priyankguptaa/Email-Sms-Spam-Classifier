Email/SMS Spam Classifier

This project aims to build a machine learning model that can accurately classify messages as spam or ham (not spam). It involves data preprocessing, exploratory analysis, model training, evaluation, and deployment via a web interface using Streamlit.


Project Workflow

1. Data Collection
    - Imported dataset from Kaggle containing labeled SMS/email messages.

2. Data Cleaning
    - Dropped irrelevant or sparse columns to reduce imbalance.
    - Renamed remaining columns for clarity.
    - Applied Label Encoding to convert categorical labels into numerical format.
    - Removed missing and duplicate entries.

3. Exploratory Data Analysis (EDA)
    - Analyzed class distribution and identified imbalance.
    - Applied techniques to balance the dataset.
    - Used NLTK to extract features like number of words, characters, and sentences.

4. Text Preprocessing
    - Converted all text to lowercase.
    - Tokenized messages into individual words.
    - Removed special characters, stop words, and punctuation.
    - Applied stemming to reduce words to their root form.

5. Visualization
    - Generated word clouds to visualize frequently used words in spam and ham messages.

6. Model Building
    - Vectorized text using both CountVectorizer and TF-IDF.
    - Trained multiple models:
        - Naive Bayes (GaussianNB, MultinomialNB, BernoulliNB)
        - Logistic Regression
        - Support Vector Classifier (SVC)
        - Decision Tree
        - K-Nearest Neighbors
        - Random Forest
        - AdaBoost
        - Bagging Classifier
        - Extra Trees Classifier
        - Gradient Boosting
        - XGBoost

7. Evaluation
    - Compared models using accuracy, precision, recall, and F1-score.
    - Found MultinomialNB with TF-IDF to be the best-performing model.

8. Improvements
    - Tuned max_features in TF-IDF to improve accuracy.
    - Tried scaling features using MinMaxScaler (no significant improvement, excluded from final pipeline).
    - Implemented a Voting Classifier combining top models for enhanced performance.

9. Deployment
    - Built a user-friendly web interface using Streamlit.
    - Users can input a message and get real-time spam classification


Technologies Used
    - Python
    - Pandas, NumPy
    - NLTK
    - Scikit-learn
    - XGBoost
    - Streamlit
    - Matplotlib
    - WordCloud

How to Run
    - Clone the repository.
    - Install dependencies.
    - Run the Streamlit app.
