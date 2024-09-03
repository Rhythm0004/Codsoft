import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('fraudTrain.csv')
print(data.head())

data = data.drop(['first', 'last', 'street', 'dob'], axis=1)

data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['trans_year'] = data['trans_date_trans_time'].dt.year
data['trans_month'] = data['trans_date_trans_time'].dt.month
data['trans_day'] = data['trans_date_trans_time'].dt.day
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data = data.drop('trans_date_trans_time', axis=1)

X = data.drop('is_fraud', axis=1)
y = data['is_fraud']
print("Value counts:", y.value_counts())

categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'trans_year', 'trans_month', 'trans_day', 'trans_hour']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10, random_state=42, criterion='gini'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("X_train", len(y_train))

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of train data:")
print(conf_matrix)

print("\n------------------------------------------------------------------------------\n")

test_data = pd.read_csv('fraudTest.csv')
print(test_data.head())

test_data = test_data.drop(['first', 'last', 'street', 'dob'], axis=1)

test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])
test_data['trans_year'] = test_data['trans_date_trans_time'].dt.year
test_data['trans_month'] = test_data['trans_date_trans_time'].dt.month
test_data['trans_day'] = test_data['trans_date_trans_time'].dt.day
test_data['trans_hour'] = test_data['trans_date_trans_time'].dt.hour
test_data = test_data.drop('trans_date_trans_time', axis=1)

y_test = test_data['is_fraud']
X_test = test_data.drop('is_fraud', axis=1)

y_test_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_test_pred))

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

