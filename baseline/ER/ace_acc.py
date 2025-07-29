import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
from codes.data import get_dynamic_loader
from codes.utils import get_model

class ACEAccuracyEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        
    def load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        model = get_model(num_classes=100, pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model = model.to(self.device)
        model.eval()
        return model
    
    def evaluate_class_wise(self, model, class_range=(0, 99)):
        """Evaluate model on each class individually and return class-wise accuracies"""
        class_accuracies = {}
        
        # Get validation loader for all classes
        val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=64)
        
        # Initialize counters for each class
        class_correct = {i: 0 for i in range(class_range[0], class_range[1] + 1)}
        class_total = {i: 0 for i in range(class_range[0], class_range[1] + 1)}
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Evaluating class-wise"):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                # Update class-wise counters
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        # Calculate class-wise accuracies
        for class_id in range(class_range[0], class_range[1] + 1):
            if class_total[class_id] > 0:
                class_accuracies[class_id] = (class_correct[class_id] / class_total[class_id]) * 100
            else:
                class_accuracies[class_id] = 0.0
        
        return class_accuracies
    
    def evaluate_overall(self, model, class_range=(0, 99)):
        """Evaluate overall accuracy on specified class range"""
        val_loader = get_dynamic_loader(class_range=class_range, mode="val", batch_size=64)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Evaluating overall"):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return accuracy
    
    def get_class_range_for_step(self, step):
        """Get class range for each step"""
        class_ranges = {
            1: (0, 59),
            2: (0, 69), 
            3: (0, 79),
            4: (0, 89),
            5: (0, 99)
        }
        return class_ranges.get(step, (0, 99))
    
    def evaluate_checkpoint(self, checkpoint_path, step_num):
        """Evaluate a single checkpoint and return results"""
        print(f"\n=== Evaluating Step {step_num} ===")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load model
        model = self.load_model(checkpoint_path)
        
        # Get class range for this step for overall accuracy
        overall_class_range = self.get_class_range_for_step(step_num)
        
        # Evaluate class-wise accuracies on all 100 classes (always 0-99)
        print("Computing class-wise accuracies...")
        class_wise_acc = self.evaluate_class_wise(model, class_range=(0, 99))
        
        # Evaluate overall accuracy on step-specific class range
        print(f"Computing overall accuracy for classes {overall_class_range[0]}-{overall_class_range[1]}...")
        overall_acc = self.evaluate_overall(model, class_range=overall_class_range)
        
        # Ensure all 100 classes are present in per_class_acc with string keys
        per_class_acc = {}
        for class_id in range(100):
            per_class_acc[str(class_id)] = class_wise_acc.get(class_id, 0.0)
        
        # Create results dictionary in the requested format
        results = {
            "overall_acc": round(overall_acc, 2),
            "per_class_acc": per_class_acc
        }
        
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        
        return results
    
    def save_results(self, results, output_path):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to: {output_path}")

def main():
    # Initialize evaluator
    evaluator = ACEAccuracyEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define checkpoint directory and results directory
    checkpoint_dir = '/home/jag/codes/Bi/baseline/ER/checkpoints/ace'
    results_dir = '/home/jag/codes/Bi/baseline/ER/results/ace'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate each step
    for step in range(1, 6):  # Steps 1-5
        checkpoint_path = os.path.join(checkpoint_dir, f'step{step}.pth')
        
        if os.path.exists(checkpoint_path):
            # Evaluate checkpoint
            results = evaluator.evaluate_checkpoint(checkpoint_path, step)
            
            # Save individual step results
            output_path = os.path.join(results_dir, f'step{step}.json')
            evaluator.save_results(results, output_path)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found!")
    
    print("All ACE results saved in:", results_dir)

if __name__ == "__main__":
    main()