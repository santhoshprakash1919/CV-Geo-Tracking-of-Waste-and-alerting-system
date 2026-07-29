import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim

def optimize_input():
    classifier_path = 'ai_model/classifier.pth'
    classifier = models.mobilenet_v2()
    classifier.classifier[1] = nn.Linear(classifier.last_channel, 3)
    classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
    classifier.eval()

    class_names = ['empty', 'full', 'partial']

    for target_idx, class_name in enumerate(class_names):
        # Create a random input tensor and enable gradients
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        optimizer = optim.Adam([x], lr=0.1)
        
        # Optimize to maximize the logit/probability of target_idx
        for step in range(50):
            optimizer.zero_grad()
            out = classifier(x)
            probs = torch.softmax(out, dim=1)[0]
            loss = -probs[target_idx] # minimize negative probability = maximize probability
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            final_out = classifier(x)
            final_probs = torch.softmax(final_out, dim=1)[0]
            print(f"Target: {class_name} | Final probs: {final_probs.tolist()} | Logits: {final_out.tolist()[0]}")

if __name__ == '__main__':
    optimize_input()
