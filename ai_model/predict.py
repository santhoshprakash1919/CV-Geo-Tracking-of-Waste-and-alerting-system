from pathlib import Path

yolo_model = None
classifier = None
transform = None
class_names = ['empty', 'full', 'partial']  # alphabetical, matches training order
_model_error = None


def _load_models():
    global yolo_model, classifier, transform, _model_error

    if _model_error is not None:
        return False, _model_error

    if yolo_model is not None and classifier is not None and transform is not None:
        return True, None

    try:
        from ultralytics import YOLO
        import torch
        from torchvision import transforms, models
        import torch.nn as nn
        from PIL import Image  # noqa: F401 - import checked here for predict()
    except ModuleNotFoundError as exc:
        _model_error = (
            'Missing ML dependency: '
            f'{exc.name}. Install ultralytics, torch, torchvision, and Pillow.'
        )
        return False, _model_error

    model_dir = Path(__file__).resolve().parent
    yolo_path = model_dir / 'best.pt'
    classifier_path = model_dir / 'classifier.pth'

    try:
        yolo_model = YOLO(str(yolo_path))
        classifier = models.mobilenet_v2()
        classifier.classifier[1] = nn.Linear(classifier.last_channel, 3)
        classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        classifier.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    except Exception as exc:
        _model_error = f'Failed to load ML models: {exc}'
        return False, _model_error

    return True, None


def _best_box(results):
    """Return the highest-confidence box from YOLO results, or None."""
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    best_i = int(boxes.conf.argmax())
    return boxes[best_i]


def predict(image_path):
    ok, error = _load_models()
    if not ok:
        return {'error': error}

    import torch
    from PIL import Image

    # 1. Try normal YOLO detection.
    results = yolo_model(image_path, conf=0.15, verbose=False)
    box = _best_box(results)

    # Fallback pass: retry once at a much lower confidence.
    low_confidence = False
    if box is None:
        results = yolo_model(image_path, conf=0.05, verbose=False)
        box = _best_box(results)
        low_confidence = True

    img = Image.open(image_path).convert('RGB')

    # Classify the whole image as reference/fallback
    whole_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        whole_output = classifier(whole_tensor)
        whole_probs = torch.softmax(whole_output, dim=1)[0]
        whole_pred_idx = torch.argmax(whole_probs).item()
        whole_pred = class_names[whole_pred_idx]
        whole_conf = whole_probs[whole_pred_idx].item()

    if box is not None:
        detection_conf = round(float(box.conf[0]), 3)
        box_coords = box.xyxy[0].tolist()
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
        if crop_pred == 'partial' and whole_pred in ['empty', 'full'] and whole_conf > 0.5:
            return {
                'prediction': whole_pred,
                'confidence': round(whole_conf, 3),
                'detection_confidence': 0.0,
                'low_confidence_detection': True
            }
        else:
            return {
                'prediction': crop_pred,
                'confidence': round(crop_conf, 3),
                'detection_confidence': detection_conf,
                'low_confidence_detection': low_confidence
            }
    else:
        # No box at all, trust whole image
        return {
            'prediction': whole_pred,
            'confidence': round(whole_conf, 3),
            'detection_confidence': 0.0,
            'low_confidence_detection': True
        }
