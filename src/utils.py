import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_prediction(model_path, model, dataset, device):

    if len(dataset.inputs) == 0:
        return "Data preprocessing failed or the file is invalid."

    model.load_state_dict(torch.load(model_path))
    model.eval()

    inputs, targets = dataset[0]
    inputs = inputs.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        predictions = model(inputs).squeeze().cpu().numpy()

    ground_truth = targets

    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, 96))
    time_steps = [
        "1 days",
        "2 days",
        "4 days",
        "7 days",
        "11 days",
        "17 days",
        "25 days",
        "37 days",
        "53 days",
        "77 days",
        "111 days",
        "158 days",
        "226 days",
        "323 days",
        "1.3 years",
        "1.8 years",
        "2.6 years",
        "3.6 years",
        "5.2 years",
        "7.3 years",
        "10.4 years",
        "14.8 years",
        "21.1 years",
        "30.0 years",
    ]

    # Visualization
    plt.figure(figsize=(30, 3 * len(time_steps)))
    for i, time_step in enumerate(time_steps):
        # Predicted saturation
        plt.subplot(len(time_steps), 2, 2 * i + 1)
        plt.pcolor(X, Y, np.flipud(predictions[:, :, i]), shading="auto")
        plt.colorbar()
        plt.xlim([0, 2000])
        plt.title(f"Predicted Gas saturation {time_step}")

        # Ground truth saturation
        plt.subplot(len(time_steps), 2, 2 * i + 2)
        plt.pcolor(X, Y, np.flipud(ground_truth[:, :, i]), shading="auto")
        plt.colorbar()
        plt.xlim([0, 2000])
        plt.title(f"Ground Truth Gas saturation {time_step}")

    plt.tight_layout()
    plt.show()
