import numpy as np
from scipy.signal import convolve2d # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore


def load_grid(filename):
    """Load grid file and convert '.' → 0, '@' → 1."""
    with open(filename, "r") as f:
        grid = [list(line.strip()) for line in f if line.strip()]
    arr = np.array(grid)
    binary = (arr == "@").astype(int)
    return binary


def compute_neighbor_counts(binary):
    """Use convolution to compute true neighbor counts."""
    kernel = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
    counts = convolve2d(binary, kernel, mode="same",
                        boundary="fill", fillvalue=0)
    return counts


def extract_training_data(binary, neighbor_counts):
    """
    Build ML dataset:
        X = flattened 3x3 patches
        y = 1 if center cell has <4 neighbors AND is '@', else 0
    """
    H, W = binary.shape
    X, y = [], []

    padded = np.pad(binary, pad_width=1, mode="constant")

    for i in range(H):
        for j in range(W):
            # Extract 3x3 patch from padded grid
            patch = padded[i:i+3, j:j+3].flatten()

            if binary[i, j] == 1:
                if neighbor_counts[i, j] < 4:
                    label = 1
                else:
                    label = 0
            else:
                label = 0

            X.append(patch)
            y.append(label)

    return np.array(X), np.array(y)


def ml_predict_less_than_four(binary, clf):
    """Use trained classifier to predict <4-neighbor '@' cells."""
    H, W = binary.shape
    padded = np.pad(binary, pad_width=1, mode="constant")
    ml_predictions = np.zeros((H, W), dtype=int)

    idx = 0
    for i in range(H):
        for j in range(W):
            patch = padded[i:i+3, j:j+3].flatten().reshape(1, -1)
            pred = clf.predict(patch)[0]
            ml_predictions[i, j] = pred
            idx += 1

    return ml_predictions


def main():
    filename = "data.txt"

    # Load grid
    binary = load_grid(filename)

    # Compute true neighbor counts
    neighbor_counts = compute_neighbor_counts(binary)

    # Building ML training set
    X, y = extract_training_data(binary, neighbor_counts)

    # Train a simple MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(12,),
                        max_iter=500,
                        random_state=0)
    clf.fit(X, y)

    # Predict using ML
    ml_pred = ml_predict_less_than_four(binary, clf)

    # Count '@' predicted to have <4 neighbors
    result = np.sum((binary == 1) & (ml_pred == 1))

    print("\n--- ML Neighbors ---")
    print(f"Number of '@' predicted with <4 neighbors: {result}")


if __name__ == "__main__":
    main()
