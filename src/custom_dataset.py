from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
