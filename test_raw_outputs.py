import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

def test_raw_outputs():
    model_dir = Path(__file__).resolve().parent / 'ai_model'
    classifier_path = model_dir / 'classifier.pth'
    
    print("Loading classifier...")
    classifier = models.mobilenet_v2()
    classifier.classifier[1] = nn.Linear(classifier.last_channel, 3)
    classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
    classifier.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = {
        'empty': 'test_empty.jpg',
        'full': 'test_full.jpg',
        'partial': 'test_partial.jpg',
    }

    class_names = ['empty', 'full', 'partial']

    for name, path in images.items():
        img = Image.open(path).convert('RGB')
        # Let's crop it like in predict.py (wait, in predict.py, it crops using YOLO. Let's just use the whole image to see raw classifier behavior, or crop using a dummy box)
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = classifier(input_tensor)
            probs = torch.softmax(out, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            print(f"Image {name}: logits={out.tolist()[0]} probs={probs.tolist()} pred={class_names[pred_idx]}")

if __name__ == '__main__':
    test_raw_outputs()
