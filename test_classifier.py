import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

def test():
    model_dir = Path(__file__).resolve().parent / 'ai_model'
    classifier_path = model_dir / 'classifier.pth'
    
    print("Loading model...")
    classifier = models.mobilenet_v2()
    classifier.classifier[1] = nn.Linear(classifier.last_channel, 3)
    try:
        classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        print("Loaded classifier successfully.")
    except Exception as e:
        print("Error loading:", e)
        return

    classifier.eval()
    
    # Test random inputs
    print("\nTesting random tensors:")
    torch.manual_seed(42)
    for i in range(5):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = classifier(x)
            probs = torch.softmax(out, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            print(f"Run {i}: logits={out.tolist()[0]} probs={probs.tolist()} pred={pred_idx}")

if __name__ == '__main__':
    test()
