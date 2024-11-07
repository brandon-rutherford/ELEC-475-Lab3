import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for training")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for training")
else:
    device = torch.device("cpu")
    print("Using CPU for training")

# Data Preparation
transform = transforms.Compose([
    transforms.Resize(224),               # Resize CIFAR-100 images to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Model Setup
alexnet = models.alexnet()
vgg16 = models.vgg16()
resnet18 = models.resnet18()

# Adjust the classifier layer to match CIFAR-100 (100 classes)
alexnet.classifier[6] = nn.Linear(4096, 100)
vgg16.classifier[6] = nn.Linear(4096, 100)
resnet18.fc = nn.Linear(512, 100)

# Move models to device
alexnet, vgg16, resnet18 = alexnet.to(device), vgg16.to(device), resnet18.to(device)

# Load pre-trained weights, ignoring the incompatible classifier layer
# Load pre-trained weights, ignoring the incompatible classifier layer
try:
    # Load pretrained weights for AlexNet, VGG16, and ResNet18
    alexnet_pretrained = torch.load('alexnet/alexnet-owt-7be5be79.pth', weights_only=True)
    alexnet_state = alexnet.state_dict()
    # Filter out the classifier layer to avoid the mismatch
    alexnet_pretrained = {k: v for k, v in alexnet_pretrained.items() if k in alexnet_state and v.size() == alexnet_state[k].size()}
    alexnet_state.update(alexnet_pretrained)
    alexnet.load_state_dict(alexnet_state, strict=False)

    vgg16_pretrained = torch.load('vgg16/vgg16-397923af.pth', weights_only=True)
    vgg16_state = vgg16.state_dict()
    vgg16_pretrained = {k: v for k, v in vgg16_pretrained.items() if k in vgg16_state and v.size() == vgg16_state[k].size()}
    vgg16_state.update(vgg16_pretrained)
    vgg16.load_state_dict(vgg16_state, strict=False)

    resnet18_pretrained = torch.load('resnet-18/resnet18-f37072fd.pth', weights_only=True)
    resnet18_state = resnet18.state_dict()
    resnet18_pretrained = {k: v for k, v in resnet18_pretrained.items() if k in resnet18_state and v.size() == resnet18_state[k].size()}
    resnet18_state.update(resnet18_pretrained)
    resnet18.load_state_dict(resnet18_state, strict=False)

except FileNotFoundError:
    print("Pretrained model parameters not found; training from scratch.")

# Loss and Optimizers
criterion = nn.CrossEntropyLoss()
optimizer_alexnet = optim.Adam(alexnet.parameters(), lr=0.001)
optimizer_vgg16 = optim.Adam(vgg16.parameters(), lr=0.001)
optimizer_resnet18 = optim.Adam(resnet18.parameters(), lr=0.001)


# Training and Validation Functions with Time Tracking
def train(model, train_loader, optimizer, epoch, model_name):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    end_time = time.time()
    avg_loss = running_loss / len(train_loader)
    elapsed_time = end_time - start_time
    print(f"[{model_name}] Epoch {epoch}: Training Loss = {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
    return avg_loss


def validate(model, val_loader, model_name):
    model.eval()
    val_loss = 0.0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    end_time = time.time()
    avg_val_loss = val_loss / len(val_loader)
    elapsed_time = end_time - start_time
    print(f"[{model_name}] Validation Loss = {avg_val_loss:.4f} | Time: {elapsed_time:.2f}s")
    return avg_val_loss


# Training Loop with Checkpoints, Time Tracking, and Periodic Plotting
if __name__ == "__main__":
    num_epochs = 50
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch} Training")

        # Train each model and validate
        train_loss_alex = train(alexnet, train_loader, optimizer_alexnet, epoch, "AlexNet")
        val_loss_alex = validate(alexnet, val_loader, "AlexNet")

        train_loss_vgg = train(vgg16, train_loader, optimizer_vgg16, epoch, "VGG16")
        val_loss_vgg = validate(vgg16, val_loader, "VGG16")

        train_loss_res = train(resnet18, train_loader, optimizer_resnet18, epoch, "ResNet18")
        val_loss_res = validate(resnet18, val_loader, "ResNet18")

        # Store losses for plotting
        train_losses.append((train_loss_alex, train_loss_vgg, train_loss_res))
        val_losses.append((val_loss_alex, val_loss_vgg, val_loss_res))

        # Save model parameters at checkpoints
        if epoch == 5:
            torch.save(alexnet.state_dict(), "alexnet_5_epochs.pth")
            torch.save(vgg16.state_dict(), "vgg16_5_epochs.pth")
            torch.save(resnet18.state_dict(), "resnet18_5_epochs.pth")
        if epoch == num_epochs:
            torch.save(alexnet.state_dict(), "alexnet_final.pth")
            torch.save(vgg16.state_dict(), "vgg16_final.pth")
            torch.save(resnet18.state_dict(), "resnet18_final.pth")

        # Plot and save separate graphs for each model every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs:
            epochs = range(1, epoch + 1)

            # AlexNet
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, [x[0] for x in train_losses], label="AlexNet Training Loss")
            plt.plot(epochs, [x[0] for x in val_losses], label="AlexNet Validation Loss", linestyle="--")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"AlexNet Training and Validation Loss (Up to Epoch {epoch})")
            plt.savefig(f"alexnet_loss_plot_epoch_{epoch}.png")
            plt.close()

            # VGG16
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, [x[1] for x in train_losses], label="VGG16 Training Loss")
            plt.plot(epochs, [x[1] for x in val_losses], label="VGG16 Validation Loss", linestyle="--")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"VGG16 Training and Validation Loss (Up to Epoch {epoch})")
            plt.savefig(f"vgg16_loss_plot_epoch_{epoch}.png")
            plt.close()

            # ResNet18
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, [x[2] for x in train_losses], label="ResNet18 Training Loss")
            plt.plot(epochs, [x[2] for x in val_losses], label="ResNet18 Validation Loss", linestyle="--")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"ResNet18 Training and Validation Loss (Up to Epoch {epoch})")
            plt.savefig(f"resnet18_loss_plot_epoch_{epoch}.png")
            plt.close()