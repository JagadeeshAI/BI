import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import backbone.vits_face as vits_face
from codes.facedata import get_class_info

def load_model(checkpoint_path, device='cuda'):
    """Load trained face recognition model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Model config from checkpoint
    model_config = {
        'loss_type': 'ArcFace',
        'GPU_ID': [0],
        'num_class': 50,
        'image_size': 224,
        'patch_size': 16,
        'ac_patch_size': 16, 
        'pad': 0,
        'dim': 192,
        'depth': 12,
        'heads': 3,
        'mlp_dim': 768,
        'pool': 'cls',
        'channels': 3,
        'dim_head': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'lora_rank': 0
    }
    
    model = vits_face.ViTs_face(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best val acc: {checkpoint['best_val_acc']:.2f}%")
    
    return model

def preprocess_image(image_path, image_size=224):
    """Preprocess single image for inference"""
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)), 
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_face(model, image_path, device='cuda'):
    """Predict face identity"""
    # Get class info
    class_info = get_class_info()
    class_names = class_info['class_names']
    
    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        # Get embeddings (inference mode)
        embeddings = model(image_tensor)
        
        # Get predictions with labels for classification
        dummy_label = torch.tensor([0]).to(device)  # Dummy label
        logits, _ = model(image_tensor, dummy_label)
        
        # Get probabilities and prediction
        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
        predicted_name = class_names[predicted_class.item()]
        confidence_score = confidence.item() * 100
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        
        print(f"\n🎯 Prediction Results:")
        print(f"Image: {image_path}")
        print(f"Predicted: {predicted_name}")
        print(f"Confidence: {confidence_score:.2f}%")
        
        print(f"\n📊 Top 5 Predictions:")
        for i in range(5):
            class_idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item() * 100
            name = class_names[class_idx]
            print(f"{i+1}. {name}: {prob:.2f}%")
        
        print(f"\n🔢 Embedding shape: {embeddings.shape}")
        print(f"Embedding norm: {torch.norm(embeddings).item():.4f}")
        
        return {
            'predicted_class': predicted_class.item(),
            'predicted_name': predicted_name,
            'confidence': confidence_score,
            'embeddings': embeddings.cpu().numpy(),
            'top5': [(class_names[idx.item()], prob.item()*100) 
                    for idx, prob in zip(top5_indices[0], top5_probs[0])]
        }

if __name__ == "__main__":
    # Test with your image
    checkpoint_path = '/home/jag/codes/Bi/checkpoints/face/best_model_arcface.pth'
    image_path = 'nanionly.jpeg'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Test prediction
    results = predict_face(model, image_path, device)