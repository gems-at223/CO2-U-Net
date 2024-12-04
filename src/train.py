import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
import os
from loss import LpLoss


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    save_dir: str = "/content/drive/My Drive/inpainting_model_UNET",
) -> None:
    lp_loss = LpLoss(d=2, p=2, size_average=True, reduction=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    plot_losses = PlotLosses()

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    for epoch in range(num_epochs):
        logs = {}
        epoch_train_losses = []
        epoch_val_losses = []

        model.train()
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lp_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = lp_loss(outputs, targets)
                epoch_val_losses.append(loss.item())

        train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        logs["loss"] = train_loss
        logs["val_loss"] = val_loss

        plot_losses.update(logs)
        plot_losses.send()

        if (epoch + 1) % 20 == 0:
            model_name = f"model_epoch_{epoch + 1}.pth"
            model_path = os.path.join(save_dir, model_name)
            torch.save(model.state_dict(), model_path)

    print("Training completed.")
