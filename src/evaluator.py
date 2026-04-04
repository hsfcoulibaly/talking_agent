import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def evaluate_pipeline(csv_file_path):
    print(f"Loading evaluation data from: {csv_file_path}\n")
    
    # Dictionary to store our TP, FP, TN, FN counts for each condition
    metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})
    
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Ensure the CSV has the correct headers
            expected_headers = ['Image_ID', 'Condition_Tested', 'Ground_Truth', 'VLM_Prediction']
            for header in expected_headers:
                if header not in reader.fieldnames:
                    raise ValueError(f"Missing required column in CSV: {header}")

            # Process each row
            for row in reader:
                condition = row['Condition_Tested']
                
                # Convert string '1' or '0' to integers
                truth = int(row['Ground_Truth'])
                pred = int(row['VLM_Prediction'])
                
                if truth == 1 and pred == 1:
                    metrics[condition]['TP'] += 1
                elif truth == 0 and pred == 0:
                    metrics[condition]['TN'] += 1
                elif truth == 0 and pred == 1:
                    metrics[condition]['FP'] += 1
                elif truth == 1 and pred == 0:
                    metrics[condition]['FN'] += 1

    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_file_path}")
        print("Please ensure your ground_truth_labels.csv is in the correct folder.")
        return

    # Now, calculate Precision, Recall, and F1 for each condition
    print("-" * 50)
    print(f"{'Condition':<20} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9}")
    print("-" * 50)

    total_f1 = 0
    valid_conditions = 0

    for condition, counts in metrics.items():
        tp = counts['TP']
        fp = counts['FP']
        fn = counts['FN']

        # Prevent division by zero errors
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        print(f"{condition:<20} | {precision:<9.2f} | {recall:<9.2f} | {f1_score:<9.2f}")
        
        total_f1 += f1_score
        valid_conditions += 1

    print("-" * 50)
    
    # Calculate the Macro-Average F1 Score across all conditions
    if valid_conditions > 0:
        macro_f1 = total_f1 / valid_conditions
        print(f"\nOVERALL MACRO F1 SCORE: {macro_f1:.2f}")
    else:
        print("\nNo valid conditions found to calculate an overall score.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "evaluation" / "ground_truth_labels.csv"
    evaluate_pipeline(str(csv_path))