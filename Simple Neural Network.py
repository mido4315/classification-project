from keras.models import Sequential
from keras.layers import Dense
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

model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

training_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"Neural Network Training Accuracy: {training_accuracy:.4f}")
print(f"Neural Network Test Accuracy: {test_accuracy:.4f}")
