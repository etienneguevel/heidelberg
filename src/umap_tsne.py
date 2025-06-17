import numpy as np
import umap

from sklearn.manifold import TSNE

translate_dict = train_dataset.info.get("label")

def get_umap(emb_dataset):
    # Unpack the array & labels
    embs = np.array([emb for emb, _ in emb_dataset])
    labels = np.array([lab for _, lab in emb_dataset])

    # Fit a model on the data
    uembs = umap.UMAP().fit_transform(embs)

    return uembs, labels

def get_tsne(emb_dataset):
    # Unpack the array & labels
    embs = np.array([emb for emb, _ in emb_dataset])
    labels = np.array([lab for _, lab in emb_dataset])

    # Fit a model on the data
    tembs = TSNE(n_components=2).fit_transform(embs)

    return tembs, labels