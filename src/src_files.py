# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Performance', axis=1)
    y = df['Performance']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_performance_distribution(df):
    sns.countplot(x='Performance', data=df)
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.show()

# model_training.py
from sklearn.svm import SVC

def train_svm(X_train, y_train):
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    return model

# model_evaluation.py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

# utils.py
import joblib

def save_model(model, filename="svm_model.pkl"):
    joblib.dump(model, filename)

def load_model(filename="svm_model.pkl"):
    return joblib.load(filename)
