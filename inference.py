import torch
from PIL import Image
from torchvision import transforms

from utils import load_crnn, ctc_greedy_decoder

# === Configuration ===
IMAGE_PATH = r"C:\Users\pasha\OneDrive\Рабочий стол\Screenshot 2025-06-27 170645.png"
CHECKPOINT_PATH = r"checkpoints\best_model.pth"
IMG_HEIGHT = 60
alphabet = "бвгджклмнпрст2456789"
num_classes = len(alphabet) + 1  # +1 for blank
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Preprocessing pipeline for a single image
def preprocess(image_path):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    new_w = max(10, int(w * IMG_HEIGHT / h))
    img = img.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1,1,H,W]
    return tensor


# Main inference function
def predict(image_path, model):
    input_tensor = preprocess(image_path)
    with torch.no_grad():
        out = model(input_tensor)  # [T,1,C]
    return ctc_greedy_decoder(out, alphabet)


if __name__ == "__main__":
    # Load model using utility
    model = load_crnn(
        CHECKPOINT_PATH,
        IMG_HEIGHT,
        num_classes,
        device=DEVICE,
        pretrained=False,
        transform="affine",
    )
    # Run prediction
    prediction, raw_output = predict(IMAGE_PATH, model)
    print(f"Prediction: {prediction}")
    print(f"Raw CTC output: {raw_output}")
