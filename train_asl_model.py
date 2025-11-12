import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm  # <-- 1. NEW IMPORT

# --- 1. Configuration & Hyperparameters ---
# (This is all safe to be in the global scope)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Configuration ---
# THIS IS THE FOLDER YOU DOWNLOADED FROM KAGGLE
DATA_DIR = './asl_alphabet_train/asl_alphabet_train/'  # Update this path if necessary

# --- Hyperparameters ---
NUM_EPOCHS = 15     # ⚠️ Set to 5 for a quick test. Increase to 15+ for high accuracy.
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMG_SIZE = 224

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Data Preprocessing & Loading ---
# (Transform definitions are safe in the global scope)

# Define the transformations
# Normalization values for ImageNet (which MobileNetV2 was trained on)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Transforms for training: include data augmentation
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),          # Resize to 224x224
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize,
])

# Transforms for validation/testing: no augmentation
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize,
])


# --- 3. Model Definition (MobileNetV2) ---
# (Function definitions are safe in the global scope)

def build_asl_model(num_classes):
    """
    Builds a transfer learning model using MobileNetV2.
    """
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all feature layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, num_classes)
        # Note: nn.CrossEntropyLoss in PyTorch includes softmax.
    )
    return model

# --- Visualization Helper ---
# (Function definitions are safe in the global scope)
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=10)
    plt.axis('off')


# --- MAIN EXECUTION FUNCTION ---
# We put all the code that *runs* inside a function
def main():
    print(f"Using device: {device}")

    # --- Check if data directory exists ---
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Please download the 'asl_alphabet_train' dataset from Kaggle")
        print("and place it in the same directory as this script.")
        exit()

    # Load the full dataset
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    # Get class names
    class_names = full_dataset.classes
    NUM_CLASSES = len(class_names)
    print(f"Found {NUM_CLASSES} classes: {class_names}")

    # Split dataset into training (80%), validation (10%), and test (10%)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Apply the correct transforms
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4  # Shuffle to get random images for viz
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = build_asl_model(NUM_CLASSES).to(device)

    # --- Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)


    # --- 4. Training Loop ---

    print("\nStarting training...")
    best_val_acc = 0.0
    BEST_MODEL_PATH = 'best_asl_model.pth'

    # --- 2. MODIFIED LOOP ---
    # Wrap the outer epoch loop with tqdm
    epoch_loop = tqdm(range(NUM_EPOCHS), desc="Total Progress")
    
    for epoch in epoch_loop:
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # --- 3. MODIFIED LOOP ---
        # Add a tqdm wrapper for the training loader
        train_loop = tqdm(train_loader, desc=f"  Epoch {epoch+1} Train", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update the inner progress bar
            train_loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            # --- 3. MODIFIED LOOP ---
            # Add a tqdm wrapper for the validation loader
            val_loop = tqdm(val_loader, desc=f"  Epoch {epoch+1} Val", leave=False)
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update the inner progress bar
                val_loop.set_postfix(loss=loss.item())

        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        # --- 4. MODIFIED PRINT ---
        # Instead of printing, update the main progress bar's description
        epoch_loop.set_postfix(
            Train_Loss=f"{train_loss:.4f}", Train_Acc=f"{train_acc:.4f}",
            Val_Loss=f"{val_loss:.4f}", Val_Acc=f"{val_acc:.4f}"
        )

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            # We can use tqdm.write to print without messing up the bars
            tqdm.write(f'New best model saved to {BEST_MODEL_PATH} (Acc: {best_val_acc:.4f})')

    print("\nTraining finished!")


    # --- 5. Visualization of 5 Random Predictions ---
    # (No changes needed here)

    print("\nLoading best model for visualization...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    try:
        inputs, labels = next(iter(test_loader))
    except StopIteration:
        print("Test loader is empty.")
        exit()

    inputs = inputs.to('cpu')
    labels = labels.to('cpu')
    model.to('cpu')

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(15, 5))
    plt.suptitle("5 Random Predictions from the Test Set", fontsize=16)

    num_to_show = min(5, BATCH_SIZE)
    if num_to_show == 0:
        print("Batch size is 0, cannot show predictions.")
        return
        
    indices = random.sample(range(len(inputs)), num_to_show)

    for i, idx in enumerate(indices):
        ax = plt.subplot(1, num_to_show, i + 1)
        
        real_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        
        title_color = 'green' if real_label == pred_label else 'red'
        
        imshow(inputs[idx], title=f"Real: {real_label}\nPredicted: {pred_label}")
        ax.title.set_color(title_color)

    plt.savefig("asl_predictions.png")
    print("\nSaved prediction plot to 'asl_predictions.png'")
    plt.show()

# --- THIS IS THE FIX ---
if __name__ == '__main__':
    main()
