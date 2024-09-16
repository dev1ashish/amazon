import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torch.cuda.amp import GradScaler, autocast

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.value_loss = nn.MSELoss(reduction='mean')
        self.unit_loss = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = alpha

    def forward(self, value_pred, unit_logits, value_target, unit_target):
        # Check for NaN or infinity values
        if torch.isnan(value_pred).any() or torch.isinf(value_pred).any():
            print("Warning: NaN or Inf in value_pred")
            value_pred = torch.nan_to_num(value_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(unit_logits).any() or torch.isinf(unit_logits).any():
            print("Warning: NaN or Inf in unit_logits")
            unit_logits = torch.nan_to_num(unit_logits, nan=0.0, posinf=1e6, neginf=-1e6)

        # Compute value loss
        value_loss = self.value_loss(value_pred, value_target)
        
        # Compute unit loss
        unit_loss = self.unit_loss(unit_logits, unit_target)
        
        # Combine losses
        total_loss = self.alpha * value_loss + (1 - self.alpha) * unit_loss
        
        return total_loss

    def compute_metrics(self, value_pred, unit_logits, value_target, unit_target):
        with torch.no_grad():
            value_accuracy = torch.mean((torch.abs(value_pred - value_target) < 0.1).float())
            _, predicted_units = torch.max(unit_logits, dim=-1)
            unit_accuracy = torch.mean((predicted_units == unit_target).float())
            combined_accuracy = torch.mean(
                ((torch.abs(value_pred - value_target) < 0.1) & (predicted_units == unit_target)).float()
            )
        return {
            'value_accuracy': value_accuracy.item(),
            'unit_accuracy': unit_accuracy.item(),
            'combined_accuracy': combined_accuracy.item()
        }

def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps):
    model.train()
    total_loss = 0
    processed_samples = 0
    
    scaler = GradScaler()
    
    progress_bar = tqdm(total=len(dataloader.dataset), desc="Training", unit="sample")
    
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        try:
            # Check if image tensor contains NaN values
            if torch.isnan(batch['image']).any():
                print(f"Warning: NaN values in image tensor in batch {i}")
                # Replace NaN values with zeros
                batch['image'] = torch.nan_to_num(batch['image'], nan=0.0)

            with autocast():
                value_pred, unit_logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['image'].to(device),
                    batch['allowed_units'].to(device)
                )
                
                loss = criterion(
                    value_pred, 
                    unit_logits, 
                    batch['value'].to(device), 
                    batch['unit'].to(device)
                )
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            processed_samples += batch['input_ids'].size(0)
            
            progress_bar.update(batch['input_ids'].size(0))
            progress_bar.set_postfix({
                'Loss': f"{total_loss / processed_samples:.4f}",
            })
            
            del value_pred, unit_logits, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"Error in batch {i}: {str(e)}")
            print("Batch data shapes:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.shape}")
            # Skip this batch and continue with the next one
            continue
    
    progress_bar.close()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            with autocast():
                value_pred, unit_logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['image'].to(device),
                    batch['allowed_units'].to(device)
                )
                
                # Ensure predictions match target shapes
                value_pred = value_pred.view(-1)
                unit_logits = unit_logits.view(batch['unit'].size(0), -1)
                
                loss = criterion(
                    value_pred, 
                    unit_logits, 
                    batch['value'].to(device), 
                    batch['unit'].to(device)
                )
            
            total_loss += loss.item()
            metrics = criterion.compute_metrics(
                value_pred, 
                unit_logits, 
                batch['value'].to(device), 
                batch['unit'].to(device)
            )
            all_metrics.append(metrics)
    
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return total_loss / len(dataloader), avg_metrics

def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate, device, batch_size=4, accumulation_steps=128):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = CustomLoss()
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, accumulation_steps)
        val_loss, val_metrics = evaluate(model, val_dataloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Metrics: {val_metrics}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mxt_model.pth')
            print("Saved best model")
        
        torch.cuda.empty_cache()
    
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return best_val_loss

def predict(model, dataset, device):
    model.eval()
    all_predictions = []
    dataloader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=4)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            with autocast():
                value_pred, unit_logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['image'].to(device),
                    batch['allowed_units'].to(device)
                )
            
            value_pred = value_pred.view(-1)
            _, predicted_units = torch.max(unit_logits, dim=-1)
            
            for v, u in zip(value_pred.cpu().numpy(), predicted_units.cpu().numpy()):
                all_predictions.append((v.item(), u.item()))
    
    return all_predictions