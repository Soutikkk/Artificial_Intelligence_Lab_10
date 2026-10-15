import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Predict a test image
index = 0
prediction = model.predict(
    x_test[index:index + 1],
    verbose=0
)

predicted_digit = np.argmax(prediction)

print("Actual Digit:", y_test[index])
print("Predicted Digit:", predicted_digit)

# Display the handwritten digit
plt.imshow(x_test[index], cmap="gray")
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis("off")
plt.show()
