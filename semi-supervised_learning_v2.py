import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import keras
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


model = keras.models.load_model('Best-Immune-Model.keras')

raw_data = pd.read_csv('rna-data/expression_data.csv')

data = pd.read_csv('rna-data/expression_data.csv', usecols=[
    '8032-S19_S19_L001', '8032-S19_S19_L002', '8032-S1_S1_L001', '8032-S1_S1_L002',
    '8032-S20_S20_L001', '8032-S20_S20_L002', '8032-S21_S21_L001', '8032-S21_S21_L002',
    '8032-S10_S10_L001', '8032-S10_S10_L002', '8032-S11_S11_L001', '8032-S11_S11_L002',
    '8032-S28_S28_L001', '8032-S28_S28_L002', '8032-S29_S29_L001', '8032-S29_S29_L002',
    '8032-S2_S2_L001', '8032-S2_S2_L002', '8032-S30_S30_L001', '8032-S30_S30_L002',
    '8032-S3_S3_L001', '8032-S3_S3_L002'
])

unlabeled_data = raw_data.drop(['8032-S19_S19_L001', '8032-S19_S19_L002', '8032-S1_S1_L001', '8032-S1_S1_L002',
                                '8032-S20_S20_L001', '8032-S20_S20_L002', '8032-S21_S21_L001', '8032-S21_S21_L002',
                                '8032-S10_S10_L001', '8032-S10_S10_L002', '8032-S11_S11_L001', '8032-S11_S11_L002',
                                '8032-S28_S28_L001', '8032-S28_S28_L002', '8032-S29_S29_L001', '8032-S29_S29_L002',
                                '8032-S2_S2_L001', '8032-S2_S2_L002', '8032-S30_S30_L001', '8032-S30_S30_L002',
                                '8032-S3_S3_L001', '8032-S3_S3_L002', 'Geneid', 'Chr', 'Start', 'End', 'Strand',
                                'Length'], axis=1)

# Extract labels (last row of each column)
labels = data.iloc[-1].values

# Remove labels from data
data = data.iloc[:-1]

# Drop features with no observed values
data = data.dropna(axis=1, how='all')
unlabeled_data = unlabeled_data.dropna(axis=1, how='all')

# Transpose data to have cells as rows and genes as columns
data = data.T
unlabeled_data = unlabeled_data.T

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
unlabeled_data_imputed = imputer.fit_transform(unlabeled_data)

# Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)
unlabeled_data_scaled = scaler.fit_transform(unlabeled_data_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

un_X_train, un_X_test = train_test_split(unlabeled_data_scaled, test_size=0.2,
                                         random_state=42)

# Reshape data for Conv1D layer (input shape: [samples, time steps, features])
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

un_X_train = un_X_train.reshape((un_X_train.shape[0], un_X_train.shape[1], 1))
un_X_test = un_X_test.reshape((un_X_test.shape[0], un_X_test.shape[1], 1))

# Print training data
print("Training Data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Define a callback to save the model with the smallest validation loss
checkpoint = ModelCheckpoint('Best-Immune-Model-semi-supervised-v2.keras', monitor='val_loss', save_best_only=True,
                             mode='min', verbose=1)

unlabeled_data_main = np.concatenate((un_X_test, un_X_train), axis=0)

# Make predictions on unlabeled data
pseudo_labels = model.predict(unlabeled_data_main)

pseudo_labels_flat = pseudo_labels.flatten()

# Combine pseudo-labels with labeled data
combined_X = np.concatenate((X_train, unlabeled_data_main), axis=0)
combined_y = np.concatenate((y_train, pseudo_labels_flat), axis=0)

# Retrain the model with combined data
history = model.fit(combined_X, combined_y, epochs=20, batch_size=64, validation_split=0.2, callbacks=[checkpoint])

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
