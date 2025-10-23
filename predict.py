import sys
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn


# =====================================================
# 1️⃣  ĐỊNH NGHĨA MODEL
# =====================================================

class Age_cnn(nn.Module):
    def __init__(self, input_channel, height, width):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

        with torch.no_grad():
            test = torch.randn(1, input_channel, height, width)
            output = self.feature(test)
            flatten_size = output.view(output.shape[0], -1).shape[1]

        self.flatten = nn.Flatten()
        self.age_fc = nn.Sequential(
            nn.Linear(flatten_size, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.flatten(x)
        return self.age_fc(x)


class Age_group_cnn(nn.Module):
    def __init__(self, input_channel, height, width):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        with torch.no_grad():
            test = torch.randn(1, input_channel, height, width)
            output = self.feature(test)
            flatten_size = output.view(output.shape[0], -1).shape[1]

        self.flatten = nn.Flatten()
        self.age_gr = nn.Sequential(
            nn.Linear(flatten_size, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 6)  # 6 nhóm tuổi
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.flatten(x)
        return self.age_gr(x)


class Gender_cnn(nn.Module):
    def __init__(self, input_channel, height, width):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15)
        )

        with torch.no_grad():
            test = torch.randn(1, input_channel, height, width)
            output = self.feature(test)
            flatten_size = output.view(output.shape[0], -1).shape[1]

        self.flatten = nn.Flatten()
        self.gender_fc = nn.Sequential(
            nn.Linear(flatten_size, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.flatten(x)
        return self.gender_fc(x)


# =====================================================
# 2️⃣  LOAD MODEL
# =====================================================

def load_model(model_class, model_path, *args):
    model = model_class(*args)
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"⚠️ Error loading model {model_path}: {e}", file=sys.stderr)
    model.eval()
    return model


# =====================================================
# 3️⃣  TIỀN XỬ LÝ ẢNH
# =====================================================

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


# =====================================================
# 4️⃣  DỰ ĐOÁN
# =====================================================

def predict_age(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        raw_value = output.item()
        if 0 <= raw_value <= 1:
            raw_value *= 100
        return int(max(0, min(100, raw_value)))


def predict_age_group(model, image_tensor):
    groups = [
        'Child (0-10)',
        'Teen (11-25)',
        'Young Adult (26-35)',
        'Adult (36-50)',
        'Middle-Aged (51-65)',
        'Senior (66-80)'
    ]
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][idx].item() * 100
    return {"group": groups[idx], "confidence": round(confidence, 2)}


def predict_gender(model, image_tensor):
    genders = ['Male', 'Female']
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][idx].item() * 100
    return {"gender": genders[idx], "confidence": round(confidence, 2)}


# =====================================================
# 5️⃣  MAIN
# =====================================================

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    image_tensor = preprocess_image(image_path)

    age_model = load_model(Age_cnn, "models/age_model.pth", 3, 128, 128)
    agegroup_model = load_model(Age_group_cnn, "models/agegroup_model.pth", 3, 128, 128)
    gender_model = load_model(Gender_cnn, "models/gender_model.pth", 3, 128, 128)

    try:
        age = predict_age(age_model, image_tensor)
    except Exception as e:
        print(f"[WARN] AgeModel failed: {e}", file=sys.stderr)
        age = 0

    age_group_result = predict_age_group(agegroup_model, image_tensor)
    gender_result = predict_gender(gender_model, image_tensor)

    result = {
        "age": age,
        "age_group": age_group_result["group"],
        "age_group_confidence": age_group_result["confidence"],
        "gender": gender_result["gender"],
        "gender_confidence": gender_result["confidence"]
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
