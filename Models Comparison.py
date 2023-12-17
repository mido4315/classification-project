from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = "/content/drive/MyDrive/heart_attack_prediction_dataset.csv"
data = pd.read_csv(file_path)

columns_to_drop = ['Patient ID', 'Blood Pressure', 'Diet', 'Country','Continent', 'Hemisphere']
data.drop(columns=columns_to_drop, inplace=True)

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

X = data.drop(columns=['Heart Attack Risk'])  # Features
y = data['Heart Attack Risk']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)

dt_clf = DecisionTreeClassifier(random_state=42)

dt_clf.fit(X_train, y_train)

dt_training_accuracy = dt_clf.score(X_train, y_train)
dt_test_accuracy = dt_clf.score(X_test, y_test)

print(dt_training_accuracy)
print(dt_test_accuracy)