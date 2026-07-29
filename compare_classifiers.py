import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
from pathlib import Path

def test_real_images():
    model_paths = {
        'workspace_classifier.pth': 'ai_model/classifier.pth',
        'downloads_classifier(3).pth': 'c:/Users/Sridharan N/Downloads/classifier (3).pth'
    }

    images = {
        'clean_00111_05': 'c:/Users/Sridharan N/Downloads/archive (2)/clean-dirty-garbage-containers-V6.1/clean-dirty-garbage-containers/train/clean/00111_05.jpg',
        'dirty_00034_01': 'c:/Users/Sridharan N/Downloads/archive (2)/clean-dirty-garbage-containers-V6.1/clean-dirty-garbage-containers/train/dirty/00034_01.jpg'
    }

    class_names = ['empty', 'full', 'partial']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for model_name, classifier_path in model_paths.items():
        print(f"\n===== Testing {model_name} =====")
        classifier = models.mobilenet_v2()
        classifier.classifier[1] = nn.Linear(classifier.last_channel, 3)
        classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        classifier.eval()

        for name, path in images.items():
            img = Image.open(path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = classifier(input_tensor)
                probs = torch.softmax(out, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                print(f"Image {name}: logits={out.tolist()[0]} probs={probs.tolist()} pred={class_names[pred_idx]}")

if __name__ == '__main__':
    test_real_images()
