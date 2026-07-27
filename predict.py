import argparse
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")

    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("checkpoint", help="Path to the checkpoint")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true")

    return parser.parse_args()


def get_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def process_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image)


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    architecture = checkpoint["architecture"]

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        input_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_features, checkpoint["hidden_units"]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(checkpoint["hidden_units"], checkpoint["output_size"])
        )
    else:
        model = models.densenet121(weights=None)
        input_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(input_features, checkpoint["hidden_units"]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(checkpoint["hidden_units"], checkpoint["output_size"])
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.to(device)
    model.eval()

    return model


def predict(image_path, model, device, top_k):
    image = process_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        probs, indices = torch.topk(probabilities, top_k)

    probs = probs.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[index] for index in indices]

    return probs, classes


def main():
    args = get_input_args()

    device = get_device(args.gpu)
    model = load_checkpoint(args.checkpoint, device)

    probabilities, classes = predict(
        args.image_path,
        model,
        device,
        args.top_k
    )

    with open(args.category_names) as file:
        cat_to_name = json.load(file)

    print("\nTop Predictions\n")

    for probability, class_id in zip(probabilities, classes):
        print(f"{cat_to_name[class_id]:30} {probability:.2%}")


if __name__ == "__main__":
    main()
