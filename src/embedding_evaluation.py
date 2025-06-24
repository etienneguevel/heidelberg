import numpy as np
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Optional
from warnings import filterwarnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

filterwarnings("ignore") 

def k_nearest_neighbor_eval(
    train_array,
    train_labels,
    test_array,
    test_labels,
    k=1,
    target_names=names
):
    # Initialize the classifier
    cls = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model
    cls.fit(train_array, train_labels)

    # Make the predictions
    preds = cls.predict(test_array)
    print(classification_report(test_labels, preds, target_names=target_names))

    return preds

def linear_probing_eval(
    train_array,
    train_labels,
    test_array,
    test_labels,
    target_names=names
):
    # Initialize the classifier
    cls = LogisticRegression()

    # Fit the model
    cls.fit(train_array, train_labels)

    # Make the predictions
    preds = cls.predict(test_array)
    print(classification_report(test_labels, preds, target_names=target_names))

    return preds

# Bonus: we can also use an MLP fitted on the embeddings to evaluate their quality.
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def mlp_eval(train_embedding, test_embedding, embedding_size, num_classes=8, target_names=names):
    # Unpack the embeddings & labels
    train_loader = DataLoader(train_embedding, batch_size=64)
    test_loader = DataLoader(test_embedding, batch_size=64)
    
    # Initialize the classifier
    hidden_size = int(embedding_size * 4/3)
    cls = Mlp(
        in_features=embedding_size,
        hidden_features=hidden_size,
        out_features=num_classes,
    )

    # Make the fitting functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cls.parameters())

    # Fit the classifier
    n_epochs = 20
    for epoch in tqdm(range(n_epochs)):
        cls.train()
        for emb, labels in train_loader:
            optimizer.zero_grad()
            outputs = cls(emb)
            labels = labels.squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Make the predictions
    label_val, preds_val = [], []
    for emb, labels in test_loader:
        outputs = cls(emb)
        preds = outputs.argmax(dim=1)

        label_val.extend(labels.to('cpu').tolist())
        preds_val.extend(preds.to('cpu').tolist())

    label_array = np.array(label_val)
    preds_array = np.array(preds_val)

    print(classification_report(label_array, preds_array, target_names=target_names))

    return preds_array

print('1-NN evaluation:\n')
_ = k_nearest_neighbor_eval(
    train_array, train_labels, test_array, test_labels, k=1
)
print('-' * 75)

print('\n20-NN evaluation:\n')
_ = k_nearest_neighbor_eval(
    train_array, train_labels, test_array, test_labels, k=20
)
print('-' * 75)

print('\nLinear probing:\n')
_ = linear_probing_eval(
    train_array, train_labels, test_array, test_labels)
print('-' * 75)

print('\nMLP trained:\n')
_ = mlp_eval(emb_train, emb_test, model.embed_dim)