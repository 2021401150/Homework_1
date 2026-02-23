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

# --- 2. CNN Model Architecture ---
class PositionPredictorCNN(nn.Module):
    def __init__(self, action_dim=4, output_dim=2):
        super(PositionPredictorCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.flattened_size = 64 * 16 * 16
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) 
        )

    def forward(self, img_before, action):
        img_features = self.conv_layers(img_before)
        img_flat = torch.flatten(img_features, start_dim=1)
        combined = torch.cat((img_flat, action), dim=1)
        predicted_pos = self.fc_layers(combined)
        return predicted_pos

# --- 3. Training Function ---
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        img_before = batch['img_before'].to(device)
        action = batch['action'].to(device)
        true_pos = batch['pos_after'].to(device)
        
        optimizer.zero_grad()
        predicted_pos = model(img_before, action)
        loss = criterion(predicted_pos, true_pos)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

# --- 4. Testing/Validation Function ---
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            img_before = batch['img_before'].to(device)
            action = batch['action'].to(device)
            true_pos = batch['pos_after'].to(device)
            
            predicted_pos = model(img_before, action)
            loss = criterion(predicted_pos, true_pos)
            
            running_loss += loss.item()
            
    return running_loss / len(dataloader)

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    output_dir = 'Second_deliverable_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Setup Data Pipeline
    full_dataset = RobotDataset('robot_data.npz')
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = PositionPredictorCNN(action_dim=4, output_dim=2).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    train_losses = []
    test_losses = []

    print("Starting CNN training...")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = test(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Test Loss: {val_loss:.6f}")

    print("Training complete!")

    # Report Final Results
    final_test_error = test_losses[-1]
    print(f"\n--- Final Results ---")
    print(f"Final CNN Test MSE: {final_test_error:.6f}")

    # Save Final Test MSE to error_output.txt
    error_txt_path = os.path.join(output_dir, 'error_output.txt')
    with open(error_txt_path, 'w') as f:
        f.write(f"Final Test Mean Squared Error (MSE): {final_test_error:.6f}\n")
    print(f"Error output saved to '{error_txt_path}'")

    # Plot the Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='x')
    
    plt.title('CNN Model: Training vs Testing Loss') 
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    
    #Save the plot inside the output folder
    plot_path = os.path.join(output_dir, 'Second_deliverable_cnn_loss_curve.png')
    plt.savefig(plot_path)
    print(f"Loss curve saved to '{plot_path}'")
    
    plt.show()

    # --- Save Predictions for Post-Analysis ---
    print("\nExtracting final predictions on the validation set...")
    model.eval() 
    all_predictions = []
    all_true_positions = []
    
    with torch.no_grad():
        for batch in val_loader:
            img_before = batch['img_before'].to(device)
            action = batch['action'].to(device)
            true_pos = batch['pos_after'].to(device)
            
            predicted_pos = model(img_before, action)
            
            all_predictions.append(predicted_pos.cpu().numpy())
            all_true_positions.append(true_pos.cpu().numpy())
            
    final_predictions_array = np.vstack(all_predictions)
    final_true_array = np.vstack(all_true_positions)
    
    # Save predictions inside the output folder
    pred_path = os.path.join(output_dir, 'Second_deliverable_predictions.npz')
    np.savez(pred_path, 
             predictions=final_predictions_array, 
             targets=final_true_array)
    print(f"Predictions saved to '{pred_path}'!")

    # Save coordinate predictions (True vs Predicted)
    coord_pred_path = os.path.join(output_dir, 'Second_deliverable_coordinate_predictions.npz')
    np.savez(coord_pred_path,
             true_pos=final_true_array,
             predicted_pos=final_predictions_array)
    print(f"Coordinate predictions saved to '{coord_pred_path}'!")

    # --- 6. Save the Trained Model ---
    model_save_path = os.path.join(output_dir, 'Second_deliverable_cnn_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model successfully saved to '{model_save_path}'!")