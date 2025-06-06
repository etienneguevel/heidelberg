import torch
from tqdm import tqdm

device = torch.device("cuda")
model.to(device)

n_epoch = 10

for epoch in range(n_epoch):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Train]")
    for inputs, labels in train_loader_tqdm:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Add a validation step
    if epoch % 5 == 0:
        model.eval()
        correct = 0
        total = 0
        val_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Valid]")
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                labels = labels.squeeze(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                acc = 100 * correct / total
                val_loader_tqdm.set_postfix(acc=f"{acc:.2f}%")
    