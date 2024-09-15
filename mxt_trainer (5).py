import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.value_loss = nn.MSELoss(reduction='mean')
        self.unit_loss = nn.CrossEntropyLoss(reduction='mean')
        self.alpha = alpha

    def forward(self, value_pred, unit_logits, value_target, unit_target):
        value_loss = self.value_loss(value_pred, value_target)
        unit_loss = self.unit_loss(unit_logits.view(-1, unit_logits.size(-1)), unit_target.view(-1))
        return self.alpha * value_loss + (1 - self.alpha) * unit_loss

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

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    processed_samples = 0
    start_time = time.time()
    
    progress_bar = tqdm(total=len(dataloader.dataset), desc="Training", unit="sample")
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        value_pred, unit_logits = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['image'].to(device),
            batch['allowed_units'].to(device)
        )
        
        loss = criterion(value_pred, unit_logits, batch['value'].to(device), batch['unit'].to(device))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        processed_samples += batch['input_ids'].size(0)
        
        # Update progress bar
        progress_bar.update(batch['input_ids'].size(0))
        progress_bar.set_postfix({
            'Loss': f"{total_loss / processed_samples:.4f}",
            'Samples/sec': f"{processed_samples / (time.time() - start_time):.2f}"
        })
    
    progress_bar.close()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            value_pred, unit_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['image'].to(device),
                batch['allowed_units'].to(device)
            )
            loss = criterion(value_pred, unit_logits, batch['value'].to(device), batch['unit'].to(device))
            total_loss += loss.item()
            metrics = criterion.compute_metrics(value_pred, unit_logits, batch['value'].to(device), batch['unit'].to(device))
            all_metrics.append(metrics)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return total_loss / len(dataloader), avg_metrics

def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate, device, batch_size=32):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = CustomLoss()
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_dataloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Metrics: {val_metrics}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_mxt_model.pth')
            print("Saved best model")
    
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return best_val_loss

def predict(model, dataset, device):
    model.eval()
    all_predictions = []
    dataloader = DataLoader(dataset, batch_size=1)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            value_pred, unit_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['image'].to(device),
                batch['allowed_units'].to(device)
            )
            _, predicted_units = torch.max(unit_logits, dim=-1)
            all_predictions.append((value_pred.item(), predicted_units.item()))
    
    return all_predictions