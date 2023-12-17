from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

svm_clf = SVC(kernel='linear', random_state=42)

svm_clf.fit(X_train, y_train)

svm_training_accuracy = svm_clf.score(X_train, y_train)
svm_test_accuracy = svm_clf.score(X_test, y_test)
print(svm_training_accuracy )
print(svm_test_accuracy)