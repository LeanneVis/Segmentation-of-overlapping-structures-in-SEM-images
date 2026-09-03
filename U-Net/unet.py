# %%
import torch
from torchvision.transforms import v2
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from utils.dataloader import dEPEDataset
from utils.model import UNet_mask
import random 
from monai.losses import DiceLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int = 42):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNG (CPU)
    torch.manual_seed(seed)

    # PyTorch RNG (all GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch 2.x compilers
    #torch.use_deterministic_algorithms(True)
    
# import mlflow

# mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
# mlflow.set_experiment("Direct-EPE")

# Dataset dEPE_sim_02
data_folder = r"C:/Users/levis/OneDrive - ASML/Documents/2602_dEPE2_AlexDuarte/depe_sim_02"


set_seed(100)

# Split Data
imgs = list(Path(data_folder).iterdir())
imgs.sort()
n_test = int(len(imgs) * 0.2) + 1
train_data = imgs[:-n_test]
test_data = imgs[-n_test:]

transforms = [
    # v2.RandomResizedCrop(200, scale=(0.4, 0.6), ratio=(0.9, 1.05), antialias=True),
    v2.RandomApply(
        transforms=[v2.RandomRotation((90, 90))],
        p=0.33,
    ),
    v2.RandomHorizontalFlip(p=0.33),
    v2.RandomVerticalFlip(p=0.33),
]

normalize_sample = True
mask = True
blur_and_noise = True
elastic_transform = False
shift = False

train_dataset = dEPEDataset(
    train_data,
    1000,
    transforms=transforms,
    mask=mask,
    normalize=normalize_sample,
    blur_and_noise=blur_and_noise,
    elastic_transform=elastic_transform,
    shift=shift
)
test_dataset = dEPEDataset(
    test_data,
    200,
    transforms=transforms,
    mask=mask,
    normalize=normalize_sample,
    blur_and_noise=blur_and_noise,
    elastic_transform=elastic_transform,
    shift=shift
    
)

batch_size = 3
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# %% Test dataloader output
iterator = iter(test_loader)
hvsem, top_sem, bottom_sem, _, _ = next(iterator)


fig, ax = plt.subplots(2, 3, figsize=(12, 4))
ax[0,0].imshow(hvsem[0, 0], cmap="gray")
ax[0,0].set_title(
    f"{hvsem.dtype} mean {torch.mean(hvsem[0]):.2f} std {torch.std(hvsem[0, 0]):.2f}"
)
ax[0,1].imshow(top_sem[0, 0], cmap="gray")
ax[0,1].set_title(
    f"{top_sem.dtype} mean {np.mean(top_sem[0].numpy()):.2f} std {np.std(top_sem[0, 0].numpy()):.2f}"
)
ax[0,2].imshow(bottom_sem[0, 0], cmap="gray")
ax[0,2].set_title(
    f"{bottom_sem.dtype} mean {np.mean(bottom_sem[0].numpy()):.2f} std {np.std(bottom_sem[0, 0].numpy()):.2f}"
)
ax[1,1].plot(top_sem[0, 0, 100, :])
ax[1,1].set_title(
    f"{top_sem.dtype} mean {np.mean(top_sem[0].numpy()):.2f} std {np.std(top_sem[0, 0].numpy()):.2f}"
)
ax[1,2].plot(bottom_sem[0, 0, 100, :])
ax[1,2].set_title(
    f"{bottom_sem.dtype} mean {np.mean(bottom_sem[0].numpy()):.2f} std {np.std(bottom_sem[0, 0].numpy()):.2f}"
)
plt.show()


# %% test trained model
# model_path = "models/2025-07-15_50_epochs_model_weights_extra_blur.pth"
# load_model = Path(model_path)

# # load saved model
# model = UNet_mask().to(device)
# model.load_state_dict(torch.load(load_model, weights_only=True))
# model.eval()

# fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# for h, t, b in test_loader:
#     h = h.to(device)
#     t = t.to(device)
#     b = b.to(device)
#     t_hat, b_hat = model(h)
#     ax[0].imshow(h.detach().cpu()[0,0], cmap='gray')
#     ax[0].axis('off')
#     ax[1].imshow(t_hat.detach().cpu()[0,0], cmap='gray')
#     ax[1].axis('off')
#     ax[2].imshow(b_hat.detach().cpu()[0,0], cmap='gray')
#     ax[2].axis('off')
# plt.show()
# %%

if __name__ == "__main__":

    
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            # torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.kaiming_normal_(
                m.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            torch.nn.init.zeros_(m.bias)

    model = UNet_mask().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Choose loss function: 'mse', 'l1', or 'hybrid'
    loss_type = 'l1'  # <-- Change this to 'mse', 'l1', or 'hybrid'
    
    dice_loss_func = DiceLoss(include_background=True, softmax=False, to_onehot_y=False)
    
    def loss_function(t, t_hat, b, b_hat, loss_type='l1'):
        """
        Flexible loss function for edge-aware training.
        
        loss_type: 'mse' (smoother), 'l1' (sharper edges), 'hybrid' (balanced)
        """
        if loss_type == 'mse':
            # Original MSE - smooth predictions
            return nn.functional.mse_loss(t_hat, t) + nn.functional.mse_loss(b_hat, b)
        
        elif loss_type == 'l1':
            # L1/MAE loss - sharper edges, less smoothing
            return nn.functional.l1_loss(t_hat, t) + nn.functional.l1_loss(b_hat, b)
        
        elif loss_type == 'hybrid':
            # Hybrid: 50% L1 + 50% MSE - balanced approach
            l1_top = nn.functional.l1_loss(t_hat, t)
            l1_bottom = nn.functional.l1_loss(b_hat, b)
            mse_top = nn.functional.mse_loss(t_hat, t)
            mse_bottom = nn.functional.mse_loss(b_hat, b)
            return 0.5 * (l1_top + l1_bottom) + 0.5 * (mse_top + mse_bottom)
        
        elif loss_type == 'dice':
                return dice_loss_func(t_hat, t) + dice_loss_func(b_hat, b) 
                
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        # return nn.functional.binary_cross_entropy_with_logits(
        #     t_hat, t
        # ) + nn.functional.binary_cross_entropy_with_logits(b_hat, b)

    def train(model, optimizer, epochs, device, loss_type='dice'):
        """Train model"""
        
        train_losses = []
        val_losses = []

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            #plt.close()
            train_loss = 0
            val_loss = 0

            model.train()
        
            for h, t, b, _, _ in tqdm(train_loader):
                h = h.to(device)
                t = t.to(device)
                b = b.to(device)
                t_hat, b_hat = model(h)
                loss = loss_function(t, t_hat, b, b_hat, loss_type=loss_type)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for h, t, b, _, _ in test_loader:
                    h = h.to(device)
                    t = t.to(device)
                    b = b.to(device)
                    t_hat, b_hat = model(h)                            
                    loss = loss_function(t, t_hat, b, b_hat, loss_type=loss_type)
                    val_loss += loss.item()

            val_loss /= len(test_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step()

            # metrics = {"train": loss, "val": val_loss}
            # mlflow.log_metrics(metrics, step=epoch)
            pbar.set_postfix(train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.4f}")

        return train_losses, val_losses

    epochs = 50
    train_losses, val_losses = train(model, optimizer, epochs=epochs, device=device, loss_type=loss_type)

    # Save model
    torch.save(
        model.state_dict(),
        f"{datetime.today().strftime('%Y-%m-%d')}_{epochs}_epochs_model_weights.pth",
    )

    
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both")

    plt.savefig("loss_curve.png")
    plt.show()


