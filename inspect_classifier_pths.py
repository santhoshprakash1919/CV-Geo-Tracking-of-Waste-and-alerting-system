import torch

def inspect_weights(path):
    print(f"\n--- Inspecting {path} ---")
    try:
        sd = torch.load(path, map_location='cpu')
        # Print shape of classifier.1.weight and bias
        if 'classifier.1.weight' in sd:
            w = sd['classifier.1.weight']
            b = sd['classifier.1.bias']
            print("classifier.1.weight shape:", w.shape)
            print("classifier.1.bias shape:", b.shape)
            print("classifier.1.bias values:", b.tolist())
        else:
            # Let's print keys containing 'classifier' or 'fc'
            keys = [k for k in sd.keys() if 'classifier' in k or 'fc' in k or 'linear' in k]
            print("Classifier/FC keys:", keys)
            for k in keys:
                print(f"{k} shape:", sd[k].shape)
    except Exception as e:
        print("Error:", e)

inspect_weights("c:/Users/Sridharan N/Downloads/classifier.pth")
inspect_weights("c:/Users/Sridharan N/Downloads/classifier (1).pth")
inspect_weights("c:/Users/Sridharan N/Downloads/classifier (2).pth")
inspect_weights("c:/Users/Sridharan N/Downloads/classifier (3).pth")
inspect_weights("c:/Users/Sridharan N/Downloads/garbage_monitoring_system_v2/garbage monitoring system/ai_model/classifier.pth")
