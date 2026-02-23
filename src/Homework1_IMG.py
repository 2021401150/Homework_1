import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os  # Added to handle directory creation

# --- 1. Dataset Class ---
class RobotDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)

        self.img_before = torch.tensor(data['img_before'], dtype=torch.float32) / 255.0
        self.img_after  = torch.tensor(data['img_after'],  dtype=torch.float32) / 255.0

        if self.img_before.shape[-1] in [3, 4]:
            self.img_before = self.img_before.permute(0, 3, 1, 2)
            self.img_after  = self.img_after.permute(0, 3, 1, 2)

        raw_actions = torch.tensor(data['action'], dtype=torch.long)
        self.action = F.one_hot(raw_actions, num_classes=4).float()

    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        return {
            'img_before': self.img_before[idx],
            'action': self.action[idx],
            'img_after': self.img_after[idx]
        }


# --- 2. Building Blocks ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ActionFiLM(nn.Module):
    def __init__(self, action_dim, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(action_dim, channels * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(channels * 2, channels * 2)
        )
        self.channels = channels

    def forward(self, feat, action):
        gamma_beta = self.fc(action)
        gamma, beta = torch.split(gamma_beta, self.channels, dim=1)
        return feat * (1.0 + gamma.view(-1, self.channels, 1, 1)) + beta.view(-1, self.channels, 1, 1)


# --- 3. U-Net Reconstructor ---
class UNetDeltaReconstructor(nn.Module):
    def __init__(self, action_dim=4, base=32, delta_scale=0.30, use_tanh=True):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.use_tanh = bool(use_tanh)

        # Encoder
        self.inc = DoubleConv(3, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))

        # Bottleneck
        self.bot = DoubleConv(base*8, base*16)
        self.film_bot = ActionFiLM(action_dim, base*16)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base*16, base*8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base*8 + base*4, base*8)
        self.film1 = ActionFiLM(action_dim, base*8)

        self.up2 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base*4 + base*2, base*4)
        self.film2 = ActionFiLM(action_dim, base*4)

        self.up3 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base*2 + base, base*2)
        self.film3 = ActionFiLM(action_dim, base*2)

        self.outc = nn.Conv2d(base*2, 3, kernel_size=1)

    def forward(self, img_before, action):
        x1 = self.inc(img_before)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        xb = self.bot(x4)
        xb = self.film_bot(xb, action)

        x = self.up1(xb)
        x = self.conv1(torch.cat([x3, x], dim=1))
        x = self.film1(x, action)

        x = self.up2(x)
        x = self.conv2(torch.cat([x2, x], dim=1))
        x = self.film2(x, action)

        x = self.up3(x)
        x = self.conv3(torch.cat([x1, x], dim=1))
        x = self.film3(x, action)

        pred_after = self.outc(x)

        if self.use_tanh:
            pred_after = torch.tanh(pred_after)
            pred_after = 0.5 * (pred_after + 1.0)

        pred_after = torch.clamp(pred_after, 0.0, 1.0)
        return pred_after


# --- 4. Training, Testing, and Saving Functions ---

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        img_in = batch['img_before'].to(device)
        act = batch['action'].to(device)
        target = batch['img_after'].to(device)

        optimizer.zero_grad()
        pred_after = model(img_in, act)
        loss = criterion(pred_after, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            img_in = batch['img_before'].to(device)
            act = batch['action'].to(device)
            target = batch['img_after'].to(device)

            pred_after = model(img_in, act)
            loss = criterion(pred_after, target)
            total_loss += loss.item()

            all_targets.append(target.cpu().numpy())
            all_preds.append(pred_after.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    return avg_loss, y_true, y_pred


def save_results(model, train_losses, val_losses, y_true, y_pred, val_loader, device, output_dir):
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save Model
    model_path = os.path.join(output_dir, 'Third_deliverable_reconstructor.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")

    # Plot Loss Graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'Third_deliverable_reconstruction_loss_curve.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Save Arrays to NPZ
    npz_path = os.path.join(output_dir, 'Third_deliverable_reconstruction_predictions.npz')
    np.savez(npz_path, targets=y_true, predictions=y_pred)
    print(f"Full predictions saved to '{npz_path}'")

    # Final Visualization
    model.eval()
    sample_batch = next(iter(val_loader))
    with torch.no_grad():
        preds = model(sample_batch['img_before'].to(device), sample_batch['action'].to(device)).cpu()

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(3):
        axes[i, 0].imshow(sample_batch['img_before'][i].permute(1, 2, 0))
        axes[i, 0].set_title("Input (Before)")
        axes[i, 1].imshow(sample_batch['img_after'][i].permute(1, 2, 0))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(preds[i].permute(1, 2, 0))
        axes[i, 2].set_title("Predicted After")
        for j in range(3):
            axes[i, j].axis('off')

    plt.tight_layout()
    sample_img_path = os.path.join(output_dir, 'Third_deliverable_sample_images.png')
    plt.savefig(sample_img_path, dpi=200)
    plt.show()


# --- 5. Main Execution Logic ---
def run_training():
    output_folder = 'Third_deliverable_outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    dataset = RobotDataset('robot_data.npz')
    train_ds, val_ds = random_split(dataset, [800, 200])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = UNetDeltaReconstructor(
        action_dim=4,
        base=32,
        delta_scale=0.30,
        use_tanh=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    print("Starting training...")
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred = test(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Final Test MSE calculation
    final_test_mse = val_losses[-1]
    print(f"\nFinal Test Mean Squared Error (MSE): {final_test_mse:.6f}")
    
    # Save the Final Test MSE to a text file
    error_file_path = os.path.join(output_folder, 'error_output.txt')
    with open(error_file_path, 'w') as f:
        f.write(f"Final Test Mean Squared Error (MSE): {final_test_mse:.6f}\n")
    print(f"Error report saved to '{error_file_path}'")

    # Save all results inside the requested folder
    save_results(model, train_losses, val_losses, y_true, y_pred, val_loader, device, output_folder)


if __name__ == "__main__":
    run_training()