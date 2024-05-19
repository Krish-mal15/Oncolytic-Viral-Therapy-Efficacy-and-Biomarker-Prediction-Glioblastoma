from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from gap_statistic import OptimalK
import numpy as np

# Load the original data
data = pd.read_csv('rna-data/expression_data.csv', usecols=[
    '8032-S19_S19_L001', '8032-S19_S19_L002', '8032-S1_S1_L001', '8032-S1_S1_L002',
    '8032-S20_S20_L001', '8032-S20_S20_L002', '8032-S21_S21_L001', '8032-S21_S21_L002',
    '8032-S10_S10_L001', '8032-S10_S10_L002', '8032-S11_S11_L001', '8032-S11_S11_L002',
    '8032-S28_S28_L001', '8032-S28_S28_L002', '8032-S29_S29_L001', '8032-S29_S29_L002',
    '8032-S2_S2_L001', '8032-S2_S2_L002', '8032-S30_S30_L001', '8032-S30_S30_L002',
    '8032-S3_S3_L001', '8032-S3_S3_L002'
])

# Extract sample IDs
sample_ids = data.index[:-1]  # Adjust to match the number of rows in data after dropping the label row

# Remove labels from data
data = data.iloc[:-1]

# Transpose data to have cells as rows and genes as columns
data = data.T

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Load the saved autoencoder model
autoencoder = load_model('Unsupervised-Biomarker-Analysis.keras')

# Extract the encoder part of the autoencoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_2').output)

# Ensure the number of features matches the input shape of the encoder
expected_features = autoencoder.input_shape[1]

if data_scaled.shape[1] != expected_features:
    raise ValueError(f"Data has {data_scaled.shape[1]} features, but model expects {expected_features} features.")

# Get latent representations using the encoder
latent_representation = encoder.predict(data_scaled)

optimalK = OptimalK(parallel_backend='joblib')
n_clusters = optimalK(latent_representation, cluster_array=np.arange(1, 10))
print('Optimal number of clusters:', n_clusters)

# Apply KMeans clustering on the latent representations
kmeans = KMeans(n_clusters=n_clusters)  # Set the number of clusters as required
clusters = kmeans.fit_predict(latent_representation)

# Visualize clustering results with annotations
plt.scatter(latent_representation[:, 0], latent_representation[:, 1], c=clusters, cmap='viridis')
for i in range(len(clusters)):
    plt.text(latent_representation[i, 0], latent_representation[i, 1], sample_ids[i], fontsize=8)
plt.title('Clustering Results (Testing Data)')
plt.xlabel('Latent Feature 1')
plt.ylabel('Latent Feature 2')
plt.colorbar(label='Cluster')
plt.show()

