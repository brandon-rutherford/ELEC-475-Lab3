import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Set device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preparation
transform = transforms.Compose([
    transforms.Resize(224),               # Resize CIFAR-100 images to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Define models and load parameters
models_dict = {
    "AlexNet": {"model": models.alexnet(), "path_5": "alexnet_5_epochs.pth", "path_final": "alexnet_final.pth"},
    "VGG16": {"model": models.vgg16(), "path_5": "vgg16_5_epochs.pth", "path_final": "vgg16_final.pth"},
    "ResNet18": {"model": models.resnet18(), "path_5": "resnet18_5_epochs.pth", "path_final": "resnet18_final.pth"}
}

# Adjust the classifier layer to match CIFAR-100 (100 classes)
for name, data in models_dict.items():
    if name == "AlexNet" or name == "VGG16":
        data["model"].classifier[6] = nn.Linear(4096, 100)
    elif name == "ResNet18":
        data["model"].fc = nn.Linear(512, 100)
    data["model"].to(device)
    data["model"].eval()  # Set model to evaluation mode

# Testing function for top-1 and top-5 errors
def test_model(model, loader):
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, top5_pred = outputs.topk(5, dim=1)

            top1_correct += (top5_pred[:, 0] == labels).sum().item()
            top5_correct += (top5_pred == labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    top1_error = 1 - top1_correct / total
    top5_error = 1 - top5_correct / total
    return top1_error, top5_error

# Test each model with both five-epoch and final parameters
for model_name, data in models_dict.items():
    print(f"\nTesting {model_name} Model")

    # Load five-epoch parameters
    data["model"].load_state_dict(torch.load(data["path_5"], map_location=device))
    top1_5, top5_5 = test_model(data["model"], test_loader)
    print(f"5-Epoch Parameters - Top-1 Error: {top1_5:.4f}, Top-5 Error: {top5_5:.4f}")

    # Load final parameters
    data["model"].load_state_dict(torch.load(data["path_final"], map_location=device))
    top1_final, top5_final = test_model(data["model"], test_loader)
    print(f"Full-Convergence Parameters - Top-1 Error: {top1_final:.4f}, Top-5 Error: {top5_final:.4f}")