import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('rna-data/expression_data.csv', usecols=[
    '8032-S19_S19_L001', '8032-S19_S19_L002', '8032-S1_S1_L001', '8032-S1_S1_L002',
    '8032-S20_S20_L001', '8032-S20_S20_L002', '8032-S21_S21_L001', '8032-S21_S21_L002',
    '8032-S10_S10_L001', '8032-S10_S10_L002', '8032-S11_S11_L001', '8032-S11_S11_L002',
    '8032-S28_S28_L001', '8032-S28_S28_L002', '8032-S29_S29_L001', '8032-S29_S29_L002',
    '8032-S2_S2_L001', '8032-S2_S2_L002', '8032-S30_S30_L001', '8032-S30_S30_L002',
    '8032-S3_S3_L001', '8032-S3_S3_L002',
])

# Extract labels (last row of each column)
labels = data.iloc[-1].values

# Remove labels from data
data = data.iloc[:-1]

# Drop features with no observed values
data = data.dropna(axis=1, how='all')

# Transpose data to have cells as rows and genes as columns
data = data.T

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

# Reshape data for Conv1D layer (input shape: [samples, time steps, features])
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Print training data
print("Training Data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Define a callback to save the model with the smallest validation loss
checkpoint = ModelCheckpoint('Best-Immune-Model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Create a more complex convolutional neural network model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

# Save the model
model.save("Immune-Response-Prediction.keras")

# Visualize training history
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
