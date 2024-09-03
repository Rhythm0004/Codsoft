import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def fileToData(file, columns):
    data = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            split_line = line.strip().split(':::')
            split_line = [element.strip() for element in split_line]
            data.append(split_line)
    df = pd.DataFrame(data, columns=columns)
    return df

data = fileToData("./train_data.txt", ['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
y = data["GENRE"]
x = (data["TITLE"].fillna('') + " " + data["DESCRIPTION"].fillna(''))

vectorizer = TfidfVectorizer()
x_transformed = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=42)
print("x_train", len(y_train))

svc = SVC(kernel='rbf', random_state=42)  
svc.fit(x_train, y_train) 

y_pred = svc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


print("\n------------------------------------------------------------------------------\n")

testData = fileToData("./test_data.txt", ['ID', 'TITLE', 'DESCRIPTION'])
x_test_test = testData["TITLE"] + " " + testData["DESCRIPTION"]
x_test_test_transformed = vectorizer.transform(x_test_test)

y_test_pred = svc.predict(x_test_test_transformed)

#----------------------------------------------------------------------------------
actualData = fileToData("./test_data_solution.txt", ['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
actual_genre = actualData["GENRE"]
actaul_data_accuracy = accuracy_score(actual_genre, y_test_pred)
print(f"Accuracy on the test data: {actaul_data_accuracy:.2f}")

testData['PREDICTED_GENRE'] = y_test_pred
testData['ACTUAL_GENRE'] = actual_genre
testData.to_csv('output/svm rbf.csv', index=False)

#-------------------------------------------------------------------------------------------

conf_matrix = confusion_matrix(actual_genre, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=svc.classes_, yticklabels=svc.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()