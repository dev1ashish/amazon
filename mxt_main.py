import os
import torch
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mxt_preprocessor import prepare_dataset, load_tokenizer
from mxt_model import load_mxt_model
from mxt_trainer import train_epoch, evaluate, CustomLoss
import pandas as pd
from tqdm import tqdm
from constants import allowed_units

# CUDA and PyTorch settings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = True

# Constants
BATCH_SIZE = 2
ACCUMULATION_STEPS = 256
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.1

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

def generate_predictions(model, dataset, device, unit_idx_to_name):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    predictions = []
    indices = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            value_pred, unit_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['image'].to(device),
                batch['allowed_units'].to(device)
            )
            
            _, predicted_units = torch.max(unit_logits, dim=-1)
            
            for v, u, idx, entity_name in zip(value_pred.cpu().numpy(), 
                                              predicted_units.cpu().numpy(), 
                                              batch['index'], 
                                              batch['entity_name']):
                v_denorm = denormalize_value(v, entity_name)
                u_name = unit_idx_to_name[u]
                predictions.append(f"{v_denorm:.2f} {u_name}")
                indices.append(idx.item())

    return indices, predictions

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
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, true in zip(predictions, dataset):
        if pred != "" and true['entity_value'] != "":
            if pred == true['entity_value']:
                true_positives += 1
            else:
                false_positives += 1
        elif pred != "" and true['entity_value'] == "":
            false_positives += 1
        elif pred == "" and true['entity_value'] != "":
            false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def main():
    print("Starting MXT model training and evaluation...")
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print_gpu_info()

    try:
        tokenizer = load_tokenizer()
        print("Tokenizer loaded successfully.")

        all_units = list(allowed_units)
        unit_to_idx = {unit: idx for idx, unit in enumerate(all_units)}
        unit_idx_to_name = {idx: unit for unit, idx in unit_to_idx.items()}
        num_units = len(all_units)
        print(f"Number of unique units: {num_units}")

        

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

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        print("Loading MXT model...")
        model = load_mxt_model(DEVICE, num_units)
        print("MXT model loaded successfully.")
        print_gpu_info()

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        criterion = CustomLoss()

        print(f"\nStarting model training for {NUM_EPOCHS} epochs...")
        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion, DEVICE, ACCUMULATION_STEPS)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss = evaluate(model, val_dataloader, criterion, DEVICE)
            print(f"Validation Loss: {val_loss:.4f}")
            print_gpu_info()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_mxt_model.pth')
                print("Saved best model")

        print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")

        print("Loading best model for predictions...")
        model.load_state_dict(torch.load('best_mxt_model.pth'))

        print("Generating predictions for test set...")
        test_indices, test_predictions = generate_predictions(model, test_dataset, DEVICE, unit_idx_to_name)

        output_df = pd.DataFrame({
            'index': test_indices,
            'prediction': test_predictions
        })
        output_file = 'test_out_mxt.csv'
        output_df.to_csv(output_file, index=False)
        print(f"Test predictions saved to {output_file}")

        print("Evaluating model on sample test set...")
        sample_test_indices, sample_test_predictions = generate_predictions(model, sample_test_dataset, DEVICE, unit_idx_to_name)
        
        sample_test_f1 = calculate_f1_score(sample_test_predictions, sample_test_dataset)
        print(f"Sample Test F1 Score: {sample_test_f1:.4f}")

        if sample_test_f1 > 0.81:
            print(f"Congratulations! Your model's F1 score ({sample_test_f1:.4f}) exceeds the current rank 1 score (0.81).")
        else:
            print(f"Your model's F1 score ({sample_test_f1:.4f}) is below the current rank 1 score (0.81). There's room for improvement!")

        print("MXT model training and evaluation completed.")

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()