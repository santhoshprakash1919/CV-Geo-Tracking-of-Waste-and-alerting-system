import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import os
import glob
import random

# Use same robust prediction function
def predict_robust(image_path, yolo_model, classifier, transform):
    class_names = ['empty', 'full', 'partial']
    
    # 1. Try normal YOLO detection
    results = yolo_model(image_path, conf=0.15, verbose=False)
    boxes = results[0].boxes
    best_box = None
    low_confidence = False
    
    if len(boxes) > 0:
        best_i = int(boxes.conf.argmax())
        best_box = boxes[best_i]
    else:
        # Try low confidence YOLO detection
        results = yolo_model(image_path, conf=0.05, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            best_i = int(boxes.conf.argmax())
            best_box = boxes[best_i]
            low_confidence = True
            
    img = Image.open(image_path).convert('RGB')
    
    # Classify whole image as reference
    whole_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        whole_output = classifier(whole_tensor)
        whole_probs = torch.softmax(whole_output, dim=1)[0]
        whole_pred_idx = torch.argmax(whole_probs).item()
        whole_pred = class_names[whole_pred_idx]
        whole_conf = whole_probs[whole_pred_idx].item()
        
    if best_box is not None:
        # Crop and classify
        box_coords = best_box.xyxy[0].tolist()
        crop = img.crop((box_coords[0], box_coords[1], box_coords[2], box_coords[3]))
        crop_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            crop_output = classifier(crop_tensor)
            crop_probs = torch.softmax(crop_output, dim=1)[0]
            crop_pred_idx = torch.argmax(crop_probs).item()
            crop_pred = class_names[crop_pred_idx]
            crop_conf = crop_probs[crop_pred_idx].item()
            
        # If crop prediction is 'partial' but whole image is highly confident 'empty' or 'full',
        # trust the whole image prediction!
        if crop_pred == 'partial' and whole_pred in ['empty', 'full'] and whole_conf > 0.8:
            return whole_pred, True # fallback
        else:
            return crop_pred, False # cropped
    else:
        return whole_pred, True # fallback

def main():
    yolo_model = YOLO('ai_model/best.pt')
    classifier = models.mobilenet_v2()
    classifier.classifier[1] = nn.Linear(classifier.last_channel, 3)
    classifier.load_state_dict(torch.load('ai_model/classifier.pth', map_location='cpu'))
    classifier.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    clean_dir = 'c:/Users/Sridharan N/Downloads/archive (2)/clean-dirty-garbage-containers-V6.1/clean-dirty-garbage-containers/train/clean'
    dirty_dir = 'c:/Users/Sridharan N/Downloads/archive (2)/clean-dirty-garbage-containers-V6.1/clean-dirty-garbage-containers/train/dirty'
    
    clean_images = glob.glob(os.path.join(clean_dir, '*.jpg'))
    dirty_images = glob.glob(os.path.join(dirty_dir, '*.jpg'))
    
    random.seed(42)
    sample_clean = random.sample(clean_images, 20)
    sample_dirty = random.sample(dirty_images, 20)
    
    print("Evaluating Clean (Expected: empty):")
    correct_clean = 0
    fallbacks_clean = 0
    for path in sample_clean:
        pred, is_fallback = predict_robust(path, yolo_model, classifier, transform)
        if pred == 'empty':
            correct_clean += 1
        if is_fallback:
            fallbacks_clean += 1
        print(f"File: {os.path.basename(path)} | Pred: {pred} | Fallback: {is_fallback}")
        
    print("\nEvaluating Dirty (Expected: full):")
    correct_dirty = 0
    fallbacks_dirty = 0
    for path in sample_dirty:
        pred, is_fallback = predict_robust(path, yolo_model, classifier, transform)
        if pred == 'full':
            correct_dirty += 1
        if is_fallback:
            fallbacks_dirty += 1
        print(f"File: {os.path.basename(path)} | Pred: {pred} | Fallback: {is_fallback}")
        
    print(f"\nClean accuracy: {correct_clean}/20 ({correct_clean*5}%) | Fallback rate: {fallbacks_clean/20:.1%}")
    print(f"Dirty accuracy: {correct_dirty}/20 ({correct_dirty*5}%) | Fallback rate: {fallbacks_dirty/20:.1%}")

if __name__ == '__main__':
    main()
