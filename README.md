# Neutral-Network-SMS-text-classifier
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Load the dataset
# This dataset is already split into train and test datasets
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/sms_spam.csv'
data = pd.read_csv(url)

# Display dataset structure
print(data.head())
print(data.info())

# Step 3: Data Preprocessing
# Encode the labels ("ham" or "spam") as 0 or 1
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['type'])  # "ham" -> 0, "spam" -> 1
X = data['text']
y = data['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the SMS messages using TfidfVectorizer (convert text into numerical data)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Step 4: Build the Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(X_train_tfidf.shape[1],), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(X_test_tfidf, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 7: Create a function to predict if a message is spam or ham
def predict_message(message):
    # Vectorize the message
    message_tfidf = vectorizer.transform([message]).toarray()
    # Predict the probability of the message being spam (1) or ham (0)
    prediction = model.predict(message_tfidf)[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    return [prediction, label]

# Testing the function
print(predict_message("Congratulations! You've won a free ticket to Bahamas!"))
print(predict_message("Hey, are we still on for dinner tonight?"))

# Step 8: Visualize training history (optional)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
