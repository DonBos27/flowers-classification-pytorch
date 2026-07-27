import argparse
import time
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_input_args():

    parser = argparse.ArgumentParser(
        description="Train a flower classifier."
    )

    parser.add_argument(
        "data_dir",
        help="Dataset directory"
    )

    parser.add_argument(
        "--save_dir",
        default="checkpoints"
    )

    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "densenet121"]
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--gpu",
        action="store_true"
    )

    return parser.parse_args()

def get_device(use_gpu):

    if use_gpu:

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")

def load_data(data_dir):

    train_dir = Path(data_dir) / "train"
    valid_dir = Path(data_dir) / "valid"
    test_dir = Path(data_dir) / "test"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transforms
    )

    valid_dataset = datasets.ImageFolder(
        valid_dir,
        transform=test_transforms
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=0
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_loader,
        valid_loader,
        test_loader
    )

def build_model(architecture, hidden_units, output_size):

    if architecture == "resnet18":

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        for parameter in model.parameters():
            parameter.requires_grad = False

        input_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, output_size)
        )

        classifier_parameters = model.fc.parameters()

    else:

        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)

        for parameter in model.parameters():
            parameter.requires_grad = False

        input_features = model.classifier.in_features

        model.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, output_size)
        )

        classifier_parameters = model.classifier.parameters()

    return model, classifier_parameters


def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    epochs
):

    best_validation_accuracy = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        running_train_loss = 0.0
        train_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            current_batch_size = images.size(0)
            running_train_loss += loss.item() * current_batch_size
            train_images += current_batch_size

        average_train_loss = running_train_loss / train_images

        model.eval()
        running_valid_loss = 0.0
        correct_predictions = 0
        valid_images = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)

                current_batch_size = images.size(0)
                running_valid_loss += loss.item() * current_batch_size
                valid_images += current_batch_size
                correct_predictions += (
                    predictions == labels
                ).sum().item()

        average_valid_loss = running_valid_loss / valid_images
        validation_accuracy = correct_predictions / valid_images
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Valid Loss: {average_valid_loss:.4f} | "
            f"Valid Accuracy: {validation_accuracy:.2%} | "
            f"Time: {epoch_time:.1f}s"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

            print(
                "Best model updated with validation accuracy: "
                f"{best_validation_accuracy:.2%}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_validation_accuracy


def save_checkpoint(
    model,
    train_dataset,
    architecture,
    hidden_units,
    learning_rate,
    epochs,
    save_dir
):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / "flower_classifier_checkpoint.pth"

    checkpoint = {
        "architecture": architecture,
        "hidden_units": hidden_units,
        "output_size": len(train_dataset.classes),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "class_to_idx": train_dataset.class_to_idx,
        "state_dict": model.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


def main():
    args = get_input_args()
    device = get_device(args.gpu)

    print(f"Using device: {device}")
    print(f"Architecture: {args.arch}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")

    (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_loader,
        valid_loader,
        test_loader
    ) = load_data(args.data_dir)

    output_size = len(train_dataset.classes)

    model, classifier_parameters = build_model(
        architecture=args.arch,
        hidden_units=args.hidden_units,
        output_size=output_size
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        classifier_parameters,
        lr=args.learning_rate
    )

    model, best_validation_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs
    )

    print(
        "Best validation accuracy: "
        f"{best_validation_accuracy:.2%}"
    )

    save_checkpoint(
        model=model,
        train_dataset=train_dataset,
        architecture=args.arch,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()