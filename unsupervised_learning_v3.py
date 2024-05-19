from keras.layers import Input, Dense, Dropout
from keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

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
sample_ids = data.index

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
x_train, x_test = train_test_split(data_scaled, test_size=0.2, random_state=42)
sample_ids_train, sample_ids_test = train_test_split(sample_ids, test_size=0.2, random_state=42)

# Define input dimension
input_dim = x_train.shape[1]

# Define encoder architecture
input_data = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_data)
encoded = Dropout(0.2)(encoded)  # Adding dropout for regularization
encoded = Dense(128, activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation='relu')(encoded)

# Define decoder architecture
decoded = Dense(128, activation='relu')(encoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Combine encoder and decoder into an autoencoder model
autoencoder = Model(input_data, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=1000, batch_size=128)

autoencoder.save('Unsupervised-Biomarker-Analysis.keras')

# Extract latent representations
encoder = Model(input_data, encoded)
latent_representation_train = encoder.predict(x_train)
latent_representation_test = encoder.predict(x_test)

# Apply clustering algorithm on the latent representations of the training data
kmeans = KMeans(n_clusters=3)
clusters_train = kmeans.fit_predict(latent_representation_train)
clusters_test = kmeans.predict(latent_representation_test)

print("Sample Ids Train: ", len(sample_ids_train))
print("Clusters Train: ", len(clusters_train))
print("Sample Ids Test: ", len(sample_ids_test))
print("Clusters Test: ", len(clusters_test))

plt.scatter(latent_representation_train[:, 0], latent_representation_train[:, 1], c=clusters_train, cmap='viridis')
for i in range(len(clusters_train)):
    plt.text(latent_representation_train[i, 0], latent_representation_train[i, 1], sample_ids_train[i], fontsize=8)
plt.title('Clustering Results (Training Data)')
plt.xlabel('Latent Feature 1')
plt.ylabel('Latent Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# Visualize clustering results for testing data with sample IDs
plt.scatter(latent_representation_test[:, 0], latent_representation_test[:, 1], c=clusters_test, cmap='viridis')
for i in range(len(clusters_test)):
    plt.text(latent_representation_test[i, 0], latent_representation_test[i, 1], sample_ids_test[i], fontsize=8)
plt.title('Clustering Results (Testing Data)')
plt.xlabel('Latent Feature 1')
plt.ylabel('Latent Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# Create folder 'unsupervised_clusters' if it doesn't exist
folder_path = 'unsupervised_clusters'
os.makedirs(folder_path, exist_ok=True)

# Save the clusters along with sample IDs to CSV files in the 'unsupervised_clusters' folder
train_cluster_df = pd.DataFrame({'Sample_ID': sample_ids_train, 'Cluster': clusters_train})
train_cluster_df.to_csv(os.path.join(folder_path, 'train_cluster_data.csv'), index=False)

test_cluster_df = pd.DataFrame({'Sample_ID': sample_ids_test, 'Cluster': clusters_test})
test_cluster_df.to_csv(os.path.join(folder_path, 'test_cluster_data.csv'), index=False)
