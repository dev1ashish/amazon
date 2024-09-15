import torch
from torch.utils.data import random_split
from mxt_preprocessor import prepare_dataset, load_tokenizer
from mxt_model import load_mxt_model
from mxt_trainer import train_model, predict
import pandas as pd
from tqdm import tqdm
from constants import allowed_units

# Constants
BATCH_SIZE = 512
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.1

def generate_predictions(model, dataset, device, unit_idx_to_name):
    predictions = predict(model, dataset, device)
    all_predictions = []
    all_indices = []
    
    for i, (value_pred, unit_pred) in enumerate(predictions):
        batch = dataset[i]
        entity_name = batch['entity_name']
        
        # Denormalize the value prediction
        v_pred_denorm = denormalize_value(value_pred, entity_name)
        u_pred_name = unit_idx_to_name[unit_pred]
        
        pred = f"{v_pred_denorm:.2f} {u_pred_name}"
        
        all_predictions.append(pred)
        all_indices.append(batch['index'])
    
    return all_indices, all_predictions

def denormalize_value(value, entity_name):
    if entity_name in ['width', 'depth', 'height', 'item_weight', 'maximum_weight_recommendation']:
        return value * 1000
    elif entity_name == 'voltage':
        return value * 1000
    elif entity_name == 'wattage':
        return value * 10000
    elif entity_name == 'item_volume':
        return value * 1000
    return value

def calculate_f1_score(predictions, dataset):
    # Implement F1 score calculation here
    # You'll need to compare predictions with actual values in the dataset
    # and calculate precision, recall, and F1 score
    # This is a placeholder function
    return 0.0

def main():
    print("Starting MXT model training and evaluation...")
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    tokenizer = load_tokenizer()
    print("Tokenizer loaded successfully.")

    # Prepare unit mapping
    all_units = list(allowed_units)
    unit_to_idx = {unit: idx for idx, unit in enumerate(all_units)}
    unit_idx_to_name = {idx: unit for unit, idx in unit_to_idx.items()}
    num_units = len(all_units)
    print(f"Number of unique units: {num_units}")

    # Prepare data
    print("Preparing datasets...")
    full_train_dataset = prepare_dataset('dataset/train.csv', 'images/train', tokenizer)
    train_size = int((1 - VAL_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    test_dataset = prepare_dataset('dataset/test.csv', 'images/test', tokenizer)
    sample_test_dataset = prepare_dataset('dataset/sample_test.csv', 'images/sample_test', tokenizer)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Sample test dataset size: {len(sample_test_dataset)}")

    # Load model
    print("Loading MXT model...")
    model = load_mxt_model(DEVICE, num_units)
    print("MXT model loaded successfully.")

    # Train model
    print(f"Starting model training for {NUM_EPOCHS} epochs...")
    best_val_loss = train_model(model, train_dataset, val_dataset, NUM_EPOCHS, LEARNING_RATE, DEVICE, BATCH_SIZE)

    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")

    # Load best model for predictions
    model.load_state_dict(torch.load('best_mxt_model.pth'))

    # Generate predictions for test set
    print("Generating predictions for test set...")
    test_indices, test_predictions = generate_predictions(model, test_dataset, DEVICE, unit_idx_to_name)

    # Create output file
    output_df = pd.DataFrame({
        'index': test_indices,
        'prediction': test_predictions
    })
    output_file = 'test_out_mxt.csv'
    output_df.to_csv(output_file, index=False)
    print(f"Test predictions saved to {output_file}")

    # Evaluate on sample test set
    print("Evaluating model on sample test set...")
    sample_test_indices, sample_test_predictions = generate_predictions(model, sample_test_dataset, DEVICE, unit_idx_to_name)
    
    # Calculate F1 score on sample test set
    sample_test_f1 = calculate_f1_score(sample_test_predictions, sample_test_dataset)
    print(f"Sample Test F1 Score: {sample_test_f1:.4f}")

    if sample_test_f1 > 0.81:
        print(f"Congratulations! Your model's F1 score ({sample_test_f1:.4f}) exceeds the current rank 1 score (0.81).")
    else:
        print(f"Your model's F1 score ({sample_test_f1:.4f}) is below the current rank 1 score (0.81). There's room for improvement!")

    print("MXT model training and evaluation completed.")

if __name__ == "__main__":
    main()