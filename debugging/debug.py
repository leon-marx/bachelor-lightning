import matplotlib.pyplot as plt
import torch
import umap
import numpy as np

if __name__ == "__main__":
    first_dim = 50
    batch_size = 32
    latent_size = 128

    latent_data = torch.randn(size=(first_dim * batch_size, latent_size))
    latent_contents = torch.randint(low=0, high=8, size=(first_dim * batch_size, 1))

    reducer = umap.UMAP(random_state=17)
    reducer.fit(latent_data)
    embedding = reducer.embedding_
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=latent_contents, cmap='gist_rainbow', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    cbar = plt.colorbar(boundaries=np.arange(2+1)-0.5)
    cbar.set_ticks(np.arange(2))
    cbar.ax.set_yticklabels([f"class_{i}" for i in range(8)])
    plt.title('UMAP projection of the latent space and normal distribution', fontsize=14)
    plt.show()