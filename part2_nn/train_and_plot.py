"""Train the from-scratch neural network and generate final Part 2 graphs."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_network import Full_NN


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "dataset.npz"
MODEL_PATH = HERE / "trained_model.npz"
HISTORY_PATH = HERE / "training_history.csv"
LOSS_PATH = HERE / "nn_loss_over_epochs.png"
PREDICTION_PATH = HERE / "nn_test_predictions.png"
PARITY_PATH = HERE / "nn_prediction_parity.png"
ERROR_PATH = HERE / "nn_error_by_joint.png"


def train(epochs=10000, hidden=(32, 32), learning_rate=0.5, seed=42):
    data = np.load(DATA_PATH)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    nn = Full_NN(X=2, HL=hidden, Y=24, seed=seed)
    history = []

    for epoch in range(1, epochs + 1):
        train_output = nn.FF(X_train)
        nn.BP(Y_train - train_output)
        nn.GD(learning_rate)

        if epoch == 1 or epoch % 100 == 0:
            train_mse = nn.mse(Y_train, nn.FF(X_train))
            test_mse = nn.mse(Y_test, nn.FF(X_test))
            history.append((epoch, train_mse, test_mse))
            print(f"Epoch {epoch:5d}/{epochs}: train MSE={train_mse:.8f}, test MSE={test_mse:.8f}")

    nn.save(MODEL_PATH)

    with HISTORY_PATH.open("w", newline="\n", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["epoch", "loss", "val_loss"])
        writer.writerows(history)

    pred_train = nn.FF(X_train)
    pred_test = nn.FF(X_test)
    return data, nn, np.asarray(history), pred_train, pred_test


def plot_loss(history):
    epochs = history[:, 0]
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, history[:, 1], linewidth=2, label="Training loss")
    plt.plot(epochs, history[:, 2], linewidth=2, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Neural Network Training and Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PATH, dpi=200)
    plt.close()


def plot_predictions(data, pred_test):
    Y_test = data["Y_test"]
    test_frame = np.arange(len(Y_test)) * 5

    plt.figure(figsize=(12, 7))
    for joint in range(6):
        plt.subplot(2, 3, joint + 1)
        plt.plot(test_frame, Y_test[:, joint], "-", linewidth=1.8, label="Target")
        plt.plot(test_frame, pred_test[:, joint], "--", linewidth=1.5, label="Prediction")
        plt.title(f"Joint {joint + 1}")
        plt.xlabel("Gait frame")
        plt.ylabel("Scaled angle")
        plt.grid(True, alpha=0.25)
        if joint == 0:
            plt.legend(fontsize=8)
    plt.suptitle("Neural Network Predictions on Test Frames", fontsize=14)
    plt.tight_layout()
    plt.savefig(PREDICTION_PATH, dpi=200)
    plt.close()


def plot_parity(Y_test, pred_test):
    actual = Y_test.ravel()
    predicted = pred_test.ravel()
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, s=8, alpha=0.35, color="#4C78A8")
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    plt.plot(limits, limits, "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Target scaled joint angle")
    plt.ylabel("Predicted scaled joint angle")
    plt.title("Test-Set Prediction Parity (24 Joints)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PARITY_PATH, dpi=200)
    plt.close()


def plot_joint_errors(Y_test, pred_test):
    rmse = np.sqrt(np.mean((Y_test - pred_test) ** 2, axis=0))
    joints = np.arange(1, 25)
    plt.figure(figsize=(12, 5.5))
    bars = plt.bar(joints, rmse, color="#59A14F")
    plt.xlabel("Joint number")
    plt.ylabel("Test RMSE (scaled angle)")
    plt.title("Test Prediction Error by Joint")
    plt.xticks(joints)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ERROR_PATH, dpi=200)
    plt.close()


def main():
    data, nn, history, pred_train, pred_test = train()
    Y_test = data["Y_test"]

    plot_loss(history)
    plot_predictions(data, pred_test)
    plot_parity(Y_test, pred_test)
    plot_joint_errors(Y_test, pred_test)

    train_mse = nn.mse(data["Y_train"], pred_train)
    test_mse = nn.mse(Y_test, pred_test)
    test_rmse = float(np.sqrt(test_mse))

    print("\nFinal results:")
    print(f"  Train MSE: {train_mse:.8f}")
    print(f"  Test MSE:  {test_mse:.8f}")
    print(f"  Test RMSE: {test_rmse:.8f}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  History: {HISTORY_PATH}")
    print(f"  Graphs: {LOSS_PATH.name}, {PREDICTION_PATH.name}, {PARITY_PATH.name}, {ERROR_PATH.name}")


if __name__ == "__main__":
    main()
