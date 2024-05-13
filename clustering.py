# resources
from data_prep_general import train
from data_prep_general import test

# modules
import matplotlib as plt
import matplotlib.pyplot as plt

#####################################
# standardscaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(train)

#k means
from sklearn.cluster import KMeans
# Choosing k after analysis or using methods like the elbow method
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# use of silhouette score to evaluate clustering (se lektionsanteckningar)
from sklearn.metrics import silhouette_score
score = silhouette_score(features_scaled, clusters)
print("Silhouette Score: ", score)

# TSNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
features_reduced = tsne.fit_transform(features_scaled)

# visualizing
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=clusters, cmap='viridis', label=clusters)
plt.title('Clustering of Gestures')
plt.colorbar(scatter)
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.show()
