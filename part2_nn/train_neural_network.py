import numpy as np 
from neural_network import Full_NN

data = np.load("part2_nn/dataset.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]

X_test = data["X_test"]
Y_test = data["Y_test"]

# create the neural network
nn = Full_NN(X=2, HL=(16,), Y=24)

# training parameters
learning_rate = 0.1
epochs = 5000

# store training loss
loss_history = []

# train the network
for epoch in range(epochs):
    predictions = nn.FF(X_train)
    error = Y_train - predictions
    nn.BP(error)
    nn.GD(learning_rate)

    loss = nn.mse(Y_train, nn.FF(X_train))
    loss_history.append(loss)

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

# test trained network
test_predictions = nn.FF(X_test)
test_loss = nn.mse(Y_test, test_predictions)

print("\nFinal Test Loss:", test_loss)

# save trained model
nn.save("part2_nn/trained_model.npz")

# save training loss
np.save("part2_nn/training_loss.npy", loss_history)

print("Training complete. Model and loss history saved.")
