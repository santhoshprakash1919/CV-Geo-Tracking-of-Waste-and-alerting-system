import torch
from PIL import Image, ImageDraw
import sys
from pathlib import Path

# Add workspace to path
sys.path.append(str(Path(__file__).resolve().parent))
from ai_model.predict import predict

def create_synthetic_images():
    print("Creating synthetic images...")
    # 1. Empty bin: solid grey color (or simple container look)
    img_empty = Image.new('RGB', (400, 400), color=(200, 200, 200))
    draw = ImageDraw.Draw(img_empty)
    # draw a simple "bin" outline
    draw.polygon([(100, 100), (300, 100), (270, 350), (130, 350)], outline=(50, 50, 50), width=5)
    img_empty.save('test_empty.jpg')

    # 2. Full bin: container filled to the top with colorful garbage shapes
    img_full = Image.new('RGB', (400, 400), color=(200, 200, 200))
    draw = ImageDraw.Draw(img_full)
    # bin outline
    draw.polygon([(100, 100), (300, 100), (270, 350), (130, 350)], outline=(50, 50, 50), width=5)
    # fill with "garbage"
    for y in range(110, 350, 20):
        for x in range(140, 260, 20):
            draw.ellipse([x-10, y-10, x+10, y+10], fill=(y % 255, x % 255, (x+y) % 255))
    img_full.save('test_full.jpg')

    # 3. Partial bin: container filled only at the bottom
    img_partial = Image.new('RGB', (400, 400), color=(200, 200, 200))
    draw = ImageDraw.Draw(img_partial)
    draw.polygon([(100, 100), (300, 100), (270, 350), (130, 350)], outline=(50, 50, 50), width=5)
    # fill only bottom half with "garbage"
    for y in range(250, 350, 20):
        for x in range(140, 260, 20):
            draw.ellipse([x-10, y-10, x+10, y+10], fill=(y % 255, x % 255, (x+y) % 255))
    img_partial.save('test_partial.jpg')

def run_tests():
    create_synthetic_images()
    
    print("\nRunning predict on synthetic images:")
    for name in ['test_empty.jpg', 'test_full.jpg', 'test_partial.jpg']:
        print(f"\n--- Predicting {name} ---")
        try:
            res = predict(name)
            print("Result:", res)
        except Exception as e:
            print("Error running prediction:", e)

if __name__ == '__main__':
    run_tests()
