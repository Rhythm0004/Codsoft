import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("./spam.csv", encoding='latin-1')[['v1', 'v2']]

data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

x = data["message"]
y = data["label"]
print("Value counts:", y.value_counts())

vectorizer = TfidfVectorizer()
x_transformed = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=42)
print("Training set size:", len(y_train))

log_reg = LogisticRegression(max_iter=1000)  
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
