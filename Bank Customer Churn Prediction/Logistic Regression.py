import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Churn_Modelling.csv')
print(data.head())

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

X = data.drop('Exited', axis=1)
y = data['Exited']
print("Value counts:", y.value_counts())

categorical_cols = ['Geography', 'Gender']
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
   ('regressor', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Training set size:", len(y_train))

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
fig.colorbar(cax)

ax.set_xticklabels([''] + ['Not Exited', 'Exited'])
ax.set_yticklabels([''] + ['Not Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
print(accuracy_score(y_test, y_pred))

for (i, j), val in np.ndenumerate(conf_matrix):
    ax.text(j, i, val, ha='center', va='center')

plt.show()
