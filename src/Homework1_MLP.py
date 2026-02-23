import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# --- 1. Dataset Class ---
class RobotDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        
        self.img_before = torch.tensor(data['img_before'], dtype=torch.float32) / 255.0
        self.img_after = torch.tensor(data['img_after'], dtype=torch.float32) / 255.0
        
        if self.img_before.shape[-1] in [1, 3, 4]: 
            self.img_before = self.img_before.permute(0, 3, 1, 2)
            self.img_after = self.img_after.permute(0, 3, 1, 2)
            
        raw_actions = torch.tensor(data['action'], dtype=torch.long)
        self.action = F.one_hot(raw_actions, num_classes=4).float()
        
        # Targets (The "True" final coordinates)
        self.pos_after = torch.tensor(data['pos_after'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.action)
    
    def __getitem__(self, idx):
        return {
            'img_before': self.img_before[idx],
            'action': self.action[idx],
            'pos_after': self.pos_after[idx],
            'img_after': self.img_after[idx]
        }

# --- 2. Model Architecture ---
class PositionPredictorMLP(nn.Module):
    def __init__(self, img_channels=3, img_height=128, img_width=128, action_dim=4, output_dim=2):
        super(PositionPredictorMLP, self).__init__()
        self.flattened_img_size = img_channels * img_height * img_width
        input_size = self.flattened_img_size + action_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) 
        )

    def forward(self, img_before, action):
        img_flat = torch.flatten(img_before, start_dim=1)
        combined_input = torch.cat((img_flat, action), dim=1)
        return self.network(combined_input)

# --- 3. Training/Testing Functions ---
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        img_before, action, true_pos = batch['img_before'].to(device), batch['action'].to(device), batch['pos_after'].to(device)
        optimizer.zero_grad()
        loss = criterion(model(img_before, action), true_pos)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            img_before, action, true_pos = batch['img_before'].to(device), batch['action'].to(device), batch['pos_after'].to(device)
            loss = criterion(model(img_before, action), true_pos)
            running_loss += loss.item()
    return running_loss / len(dataloader)

# --- 4. Main Execution ---
if __name__ == "__main__":
    output_dir = 'First_deliverable_outputs'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    full_dataset = RobotDataset('robot_data.npz')
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PositionPredictorMLP().to(device)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = [], []
    for epoch in range(20):
        t_loss = train(model, train_loader, criterion, optimizer, device)
        v_loss = test(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        print(f"Epoch [{epoch+1}/20] | Train Loss: {t_loss:.6f} | Test Loss: {v_loss:.6f}")

    print(f"Final Test Mean Squared Error (MSE): {test_losses[-1]:.6f}")
    # Save Error Output
    with open(os.path.join(output_dir, 'error_output.txt'), 'w') as f:
        f.write(f"Final Test Mean Squared Error (MSE): {test_losses[-1]:.6f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train'); plt.plot(test_losses, label='Test')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'First_deliverable_mlp_loss_curve.png'))

    # --- Save Predictions and True Coordinates ---
    model.eval()
    all_preds, all_true_values = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            img_before, action = batch['img_before'].to(device), batch['action'].to(device)
            preds = model(img_before, action)
            
            all_preds.append(preds.cpu().numpy())
            all_true_values.append(batch['pos_after'].numpy())
            
    final_preds = np.vstack(all_preds)
    final_true = np.vstack(all_true_values)

    # Standard predictions file
    np.savez(os.path.join(output_dir, 'First_deliverable_predictions.npz'), 
             predictions=final_preds, targets=final_true)

    # Updated file storing True Values and Predicted Values for comparison
    np.savez(os.path.join(output_dir, 'First_deliverable_coordinate_predictions.npz'), 
             true_pos=final_true, 
             predicted_pos=final_preds)
             
    print(f"Coordinate predictions (True vs Predicted) saved to {output_dir}/First_deliverable_coordinate_predictions.npz")

    torch.save(model.state_dict(), os.path.join(output_dir, 'First_deliverable_mlp_model.pth'))