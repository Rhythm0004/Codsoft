import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def fileToData(file):
    data = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            split_line = line.strip().split(':::')
            split_line = [element.strip() for element in split_line]
            data.append(split_line)
    df = pd.DataFrame(data, columns=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
    return df

data = fileToData("./train_data.txt")
y = data["GENRE"]
x = data["TITLE"] + " " + data["DESCRIPTION"]

vectorizer = TfidfVectorizer()
x_transformed = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.5, random_state=42)
print("x_train", len(y_train))

svc = SVC(kernel='poly', random_state=42) 
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))