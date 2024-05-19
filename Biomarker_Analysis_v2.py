from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from gap_statistic import OptimalK
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

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
sample_ids = data.columns

# Extract progression status (assume it's the last row)
progression_status = data.iloc[-1]
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

# Determine optimal number of clusters
optimalK = OptimalK(parallel_backend='joblib')
n_clusters = optimalK(latent_representation, cluster_array=np.arange(1, 10))
print('Optimal number of clusters:', n_clusters)

# Apply KMeans clustering on the latent representations
kmeans = KMeans(n_clusters=9)
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

# Create a DataFrame for cluster and progression status
cluster_progression = pd.DataFrame({
    'Cluster': clusters,
    'Progression_Status': progression_status.values
}, index=data.index)

# Use original data (without imputation) for differential expression analysis
original_gene_expression_data = pd.DataFrame(data, index=data.index, columns=data.columns)

# Identify differentially expressed genes between clusters that correlate with progression status
cluster1_genes = original_gene_expression_data.loc[cluster_progression['Cluster'] == 0]
cluster2_genes = original_gene_expression_data.loc[cluster_progression['Cluster'] == 1]

# Filter out genes with no variance to avoid precision loss in t-test
variance_threshold = 1e-5
filtered_genes = original_gene_expression_data.columns[
    original_gene_expression_data.var() > variance_threshold
]

# Perform t-tests to find differentially expressed genes
p_values = []
for gene in filtered_genes:
    t_stat, p_val = ttest_ind(cluster1_genes[gene], cluster2_genes[gene], nan_policy='omit')
    p_values.append(p_val)

# Adjust for multiple testing (e.g., using Bonferroni correction)
adjusted_p_values = multipletests(p_values, method='bonferroni')[1]

# Identify significant genes
significant_genes = filtered_genes[adjusted_p_values < 0.1]
if len(significant_genes) == 0:
    print("No significant genes found with adjusted threshold.")
else:
    # Create a heatmap for the significant genes
    plt.figure(figsize=(10, 8))
    if original_gene_expression_data[significant_genes].var().max() > 0:
        sns.clustermap(original_gene_expression_data[significant_genes], col_cluster=False, cmap='viridis')
    else:
        print("All gene expression values are identical. Adjusting colormap for visualization.")
        sns.clustermap(original_gene_expression_data[significant_genes], col_cluster=False, cmap='magma')
    plt.title('Differentially Expressed Genes')
    plt.show()
