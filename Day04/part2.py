import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors
import time

# ==============================================================================
# 1. GPU SETUP & MODEL
# ==============================================================================


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"✅ GPU DETECTED: {name}")
        # Print VRAM in GB
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM: {vram:.2f} GB")

        # RTX 30-series optimization
        torch.backends.cudnn.benchmark = True
        return d
    else:
        print("⚠️ GPU NOT DETECTED. Running on CPU.")
        return torch.device("cpu")


class RuleLearner(nn.Module):
    def __init__(self):
        super().__init__()
        # Increased model size (64 neurons) to better utilize the 3050
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 2. THE SCHOOL
# ==============================================================================


def create_training_data(num_samples=2000000, device="cpu"):
    """
    Generates millions of random 3x3 scenarios directly on the GPU.
    """
    print(f"   ⚡ Generating {num_samples:,} examples directly on {device}...")

    # Generate random 0 and 1 directly on GPU VRAM
    X = torch.randint(0, 2, (num_samples, 3, 3), device=device).float()

    center = X[:, 1, 1]

    # Sum of neighbors (Sum of all 9 minus the center)
    neighbor_sum = X.sum(dim=(1, 2)) - center

    # Rule: Remove ifnumber of Neighbors < 4
    should_remove = (center == 1) & (neighbor_sum < 4)

    # Flatten input: [N, 9]
    X_flat = X.view(num_samples, 9)
    y = should_remove.float().view(-1, 1)

    return X_flat, y

# ==============================================================================
# 3. TRAIN LOOP
# ==============================================================================


def train_model(device):
    print(f"\n🧠 Initializing Nano-Brain on {device}...")

    model = RuleLearner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.BCELoss()

    X, y = create_training_data(num_samples=9000000, device=device)

    print(f"   Training on {len(X):,} samples...")
    start_time = time.time()

    for epoch in range(1001):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            acc = ((pred > 0.5) == (y > 0.5)).float().mean()
            print(
                f"   Epoch {epoch:4d}: Loss {loss.item():.5f} | Accuracy: {acc*100:.4f}%")

            if acc > 0.99995:
                break

    print(f"✅ Training finished in {time.time() - start_time:.2f}s.\n")

    del X, y
    torch.cuda.empty_cache()

    return model

# ==============================================================================
# 4. SOLVING
# ==============================================================================


def load_grid(filename):
    with open(filename, "r") as f:
        lines = [list(line.strip()) for line in f if line.strip()]
    arr = np.array(lines)
    return (arr == "@").astype(np.float32)


def solve_on_gpu(grid_np, model, device):
    H, W = grid_np.shape
    total_removed = 0
    step = 0

    # Initialize heatmap on CPU
    heatmap_cpu = np.zeros((H, W), dtype=int)

    # Move the MAIN grid to GPU VRAM
    grid = torch.tensor(grid_np, dtype=torch.float32, device=device)

    print("⚡ Starting GPU Inference Loop...")

    while True:
        # 1. Create Sliding Windows using Unfold
        img_batch = grid.unsqueeze(0).unsqueeze(0)

        # Extracts 3x3 patches. Output: [1, 9, H*W]
        patches = F.unfold(img_batch, kernel_size=3, padding=1)
        patches_flat = patches.permute(0, 2, 1).squeeze(0)

        # 2. Ask the Neural Net
        with torch.no_grad():
            preds = model(patches_flat)

        # 3. Reshape back to [H, W] mask
        remove_prob = preds.view(H, W)

        # 4. Determine removal
        to_remove_mask = (remove_prob > 0.5) & (grid == 1)

        count = to_remove_mask.sum().item()
        total_removed += count

        if step % 1 == 0:
            print(f"   Step {step}: GPU removed {int(count)} cells")

        if count == 0:
            break

        mask_cpu = to_remove_mask.cpu().numpy()
        heatmap_cpu[mask_cpu] = step + 1

        grid[to_remove_mask] = 0.0
        step += 1

    return grid.cpu().numpy(), heatmap_cpu, total_removed

# ==============================================================================
# MAIN
# ==============================================================================


def main():
    device = get_device()

    # 1. Load Data
    try:
        grid_np = load_grid("data.txt")
    except FileNotFoundError:
        print("❌ Error: data.txt not found.")
        return

    print(
        f"📄 Loaded Grid: {grid_np.shape} | Initial Paper: {int(grid_np.sum())}")

    # 2. Train
    model = train_model(device)

    # 3. Solve
    final_grid, heatmap, total = solve_on_gpu(grid_np, model, device)

    print("-" * 40)
    print(f"🏁 Final Result: {int(total)} cells removed.")
    print(f"   Remaining: {int(final_grid.sum())}")
    print("-" * 40)

    # 4. Visualize
    print("🎨 Generating Heatmap...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    cmap = plt.get_cmap('inferno')
    cmap.set_bad(color='black')
    masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
    im = ax[0].imshow(masked_heatmap, cmap=cmap)
    ax[0].set_title(f"Removal Heatmap (Total: {int(total)})")
    plt.colorbar(im, ax=ax[0])

    # Final State
    cmap_binary = colors.ListedColormap(['black', 'white'])
    ax[1].imshow(final_grid, cmap=cmap_binary)
    ax[1].set_title("Remaining Paper Rolls")

    plt.show()


if __name__ == "__main__":
    main()
