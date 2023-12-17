from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

file_path = "/content/drive/MyDrive/heart_attack_prediction_dataset.csv"
data = pd.read_csv(file_path)

columns_to_drop = ['Patient ID', 'Blood Pressure', 'Diet', 'Country', 'Continent', 'Hemisphere']
data.drop(columns=columns_to_drop, inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

X = data.drop(columns=['Heart Attack Risk'])  
y = data['Heart Attack Risk'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train, y_train)

training_accuracy = dt_classifier.score(X_train, y_train)
test_accuracy = dt_classifier.score(X_test, y_test)

print(f"Decision Tree Training Accuracy: {training_accuracy:.4f}")
print(f"Decision Tree Test Accuracy: {test_accuracy:.4f}")
