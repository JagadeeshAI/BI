import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
from codes.data import get_dynamic_loader
from codes.utils import get_model

class EWCAccuracyEvaluator:
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
    
    def evaluate_checkpoint(self, checkpoint_path, step_num):
        """Evaluate a single checkpoint and return results"""
        print(f"\n=== Evaluating Step {step_num} ===")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load model
        model = self.load_model(checkpoint_path)
        
        # Evaluate class-wise accuracies on all 100 classes
        print("Computing class-wise accuracies...")
        class_wise_acc = self.evaluate_class_wise(model, class_range=(0, 99))
        
        # Evaluate overall accuracy on all 100 classes
        print("Computing overall accuracy...")
        overall_acc = self.evaluate_overall(model, class_range=(0, 99))
        
        # Create results dictionary
        results = {
            "step": step_num,
            "checkpoint_path": checkpoint_path,
            "overall_accuracy": overall_acc,
            "class_wise_accuracies": class_wise_acc,
            "statistics": {
                "mean_accuracy": sum(class_wise_acc.values()) / len(class_wise_acc),
                "min_accuracy": min(class_wise_acc.values()),
                "max_accuracy": max(class_wise_acc.values()),
                "num_classes": len(class_wise_acc),
                "classes_above_50": sum(1 for acc in class_wise_acc.values() if acc > 50.0),
                "classes_above_70": sum(1 for acc in class_wise_acc.values() if acc > 70.0),
                "classes_above_90": sum(1 for acc in class_wise_acc.values() if acc > 90.0),
            }
        }
        
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        print(f"Mean Class-wise Accuracy: {results['statistics']['mean_accuracy']:.2f}%")
        print(f"Classes above 50%: {results['statistics']['classes_above_50']}/100")
        print(f"Classes above 70%: {results['statistics']['classes_above_70']}/100")
        
        return results
    
    def save_results(self, results, output_path):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

def main():
    # Initialize evaluator
    evaluator = EWCAccuracyEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define checkpoint directory and results directory
    checkpoint_dir = '/home/jag/codes/Bi/baseline/ER/checkpoints/ewc'
    results_dir = '/home/jag/codes/Bi/baseline/ER/results/ewc'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate each step
    all_results = {}
    
    for step in range(1, 6):  # Steps 1-5
        checkpoint_path = os.path.join(checkpoint_dir, f'step{step}.pth')
        
        if os.path.exists(checkpoint_path):
            # Evaluate checkpoint
            results = evaluator.evaluate_checkpoint(checkpoint_path, step)
            
            # Save individual step results
            output_path = os.path.join(results_dir, f'step{step}.json')
            evaluator.save_results(results, output_path)
            
            # Store in all results
            all_results[f'step{step}'] = results
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found!")
    
    # Save combined results
    combined_output_path = os.path.join(results_dir, 'ewc_all_steps.json')
    evaluator.save_results(all_results, combined_output_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for step_key, results in all_results.items():
        step_num = results['step']
        overall_acc = results['overall_accuracy']
        mean_class_acc = results['statistics']['mean_accuracy']
        classes_above_70 = results['statistics']['classes_above_70']
        
        print(f"Step {step_num}:")
        print(f"  Overall Accuracy: {overall_acc:.2f}%")
        print(f"  Mean Class Accuracy: {mean_class_acc:.2f}%")
        print(f"  Classes >70%: {classes_above_70}/100")
        print()
    
    print("All results saved in:", results_dir)

if __name__ == "__main__":
    main()