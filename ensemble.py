import torch
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-100 test dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


# Function to load models for a specific training phase
def load_models(epoch_type):
    model_paths = {
        "5_epoch": {"AlexNet": "alexnet_5_epochs.pth", "VGG16": "vgg16_5_epochs.pth", "ResNet18": "resnet18_5_epochs.pth"},
        "full": {"AlexNet": "alexnet_final.pth", "VGG16": "vgg16_final.pth", "ResNet18": "resnet18_final.pth"}
    }

    models_dict = {
        "AlexNet": {"model": models.alexnet(), "path": model_paths[epoch_type]["AlexNet"]},
        "VGG16": {"model": models.vgg16(), "path": model_paths[epoch_type]["VGG16"]},
        "ResNet18": {"model": models.resnet18(), "path": model_paths[epoch_type]["ResNet18"]}
    }

    # Modify classifier layers to match CIFAR-100 classes
    models_dict["AlexNet"]["model"].classifier[6] = torch.nn.Linear(4096, 100)
    models_dict["VGG16"]["model"].classifier[6] = torch.nn.Linear(4096, 100)
    models_dict["ResNet18"]["model"].fc = torch.nn.Linear(512, 100)

    # Load model weights and move to device
    for model_info in models_dict.values():
        model_info["model"].load_state_dict(torch.load(model_info["path"], map_location=device))
        model_info["model"].to(device)
        model_info["model"].eval()

    return [model_info["model"] for model_info in models_dict.values()]


# Ensemble methods
def ensemble_max_probability(models, images):
    probabilities = [F.softmax(model(images), dim=1) for model in models]
    max_probs = torch.max(torch.stack(probabilities), dim=0)[0]
    preds = torch.argmax(max_probs, dim=1)
    return preds


def ensemble_probability_averaging(models, images):
    probabilities = [F.softmax(model(images), dim=1) for model in models]
    avg_probs = torch.mean(torch.stack(probabilities), dim=0)
    preds = torch.argmax(avg_probs, dim=1)
    return preds


def ensemble_majority_voting(models, images):
    outputs = [torch.argmax(model(images), dim=1) for model in models]
    stacked_outputs = torch.stack(outputs)
    preds, _ = torch.mode(stacked_outputs, dim=0)
    return preds


# Testing function
def test_ensemble_method(ensemble_method, models, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = ensemble_method(models, images)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    error_rate = 100 - accuracy  # Calculating error rate
    return error_rate


# Dictionary to store results for the table
results = {"Method": [], "# Epochs": [], "Top-1 Error Rate": []}

# Test each ensemble method for both 5-epoch and full convergence models
for epoch_type in ["5_epoch", "full"]:
    models_list = load_models(epoch_type)

    # Maximum Probability Ensemble
    max_prob_error = test_ensemble_method(ensemble_max_probability, models_list, test_loader)
    results["Method"].append("Maximum Probability")
    results["# Epochs"].append("5" if epoch_type == "5_epoch" else "Full")
    results["Top-1 Error Rate"].append(max_prob_error)

    # Probability Averaging Ensemble
    avg_prob_error = test_ensemble_method(ensemble_probability_averaging, models_list, test_loader)
    results["Method"].append("Average Probability")
    results["# Epochs"].append("5" if epoch_type == "5_epoch" else "Full")
    results["Top-1 Error Rate"].append(avg_prob_error)

    # Majority Voting Ensemble
    majority_vote_error = test_ensemble_method(ensemble_majority_voting, models_list, test_loader)
    results["Method"].append("Majority Voting")
    results["# Epochs"].append("5" if epoch_type == "5_epoch" else "Full")
    results["Top-1 Error Rate"].append(majority_vote_error)

# Displaying the results in table form
import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)