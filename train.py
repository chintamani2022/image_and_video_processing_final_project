import torch.nn.functional as F
import torch

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device):
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss.item():.4f}")

        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1} Accuracy: {accuracy:.2f}%")
