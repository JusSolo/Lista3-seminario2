from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, SpectralEmbedding, MDS
import matplotlib.pyplot as plt

X, color = make_s_curve(n_samples=3000)

# Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X)

# LLE
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Spectral Embedding
spectral = SpectralEmbedding(n_components=2)
X_spectral = spectral.fit_transform(X)

# MDS
mds = MDS(n_components=2)
X_mds = mds.fit_transform(X)

# Plot
fig = plt.figure(figsize=(10, 8))

# Isomap
plt.subplot(231)
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color)
plt.title('Isomap')

# LLE
plt.subplot(232)
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=color)
plt.title('LLE')

# t-SNE
plt.subplot(233)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color)
plt.title('t-SNE')

# Spectral Embedding
plt.subplot(234)
plt.scatter(X_spectral[:, 0], X_spectral[:, 1], c=color)
plt.title('Spectral Embedding')

# MDS
plt.subplot(235)
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=color)
plt.title('MDS')

ax = fig.add_subplot(236, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.viridis, s=10)
ax.set_title('Datos Originales')

plt.tight_layout()
plt.show()
