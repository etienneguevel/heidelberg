import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):
    def __init__(self, dataset, model, transform, device=torch.device('cuda')):
        self.device = device
        self.transform = transform
        self.embeddings, self.labels = self._create_vectors(model, dataset)

    def _create_vectors(self, model, dataset):
        embeddings = []
        label_list = []
        model.to(self.device)

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        # Loop over the data
        for batch in tqdm(loader, desc="Calculating embeddings", unit="batch"):
            # Load and transform the images
            images, labels = batch
            if self.transform:
                images = self.transform(images)
            
            images = images.to(self.device)

            # Make the embeddings from the batch
            with torch.no_grad():
                embs = model(images).to("cpu")
            
            embeddings.append(embs)
            label_list.append(labels)

        # Return the calculated embeddings
        embeddings = torch.cat(embeddings, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embeddings, label_list

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx, :], self.labels[idx]
