import tensorflow as tf
import numpy as np

true_m = 12.5
true_c = 25

NUM_SAMPLES = 100
X_values = np.linspace(0, 10, NUM_SAMPLES)
Y_values = true_m * X_values + true_c + np.random.randn(NUM_SAMPLES) * 1.5

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("Starting model training...")
history = model.fit(X_values, Y_values, epochs=50, verbose=1)
print("Training finished.")

m_learned = model.layers[0].get_weights()[0][0][0]
c_learned = model.layers[0].get_weights()[1][0]

print(f"Original m: {true_m:.2f}, c: {true_c:.2f}")
print(f"Learned m: {m_learned:.2f}, c: {c_learned:.2f}")

x_new = np.array([15.0])
y_predicted = model.predict(x_new)

print(f"\nPrediction for x = {x_new[0]:.0f}:")
print(f"Predicted y = {y_predicted[0][0]:.2f}")