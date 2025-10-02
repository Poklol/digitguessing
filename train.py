import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

# Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the pixel values
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,        # Random rotation
    width_shift_range=0.15,   # Random horizontal shift
    height_shift_range=0.15,  # Random vertical shift
    zoom_range=0.15,         # Random zoom
    shear_range=10,          # Random shear
    fill_mode='constant',    # Fill with black (0)
    cval=0                   # Black fill value
)

# Fit the data generator on training data
datagen.fit(train_images)

# Create an improved CNN model
model = models.Sequential([
    # First convolution block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second convolution block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third convolution block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Print model summary
model.summary()

# Compile with a slightly lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Add callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,           # More patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    verbose=1
)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=128),
    epochs=20,  # Train for longer
    steps_per_epoch=len(train_images) // 128,
    validation_data=(test_images, test_labels),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print(f'\nTest accuracy: {test_acc:.4f}')

# Save training history
history_path = Path("models/training_history.npz")
np.savez(history_path,
         acc=history.history['accuracy'],
         val_acc=history.history['val_accuracy'],
         loss=history.history['loss'],
         val_loss=history.history['val_loss'])

# Save the model
Path("models").mkdir(parents=True, exist_ok=True)
model.save("models/mnist_cnn.keras")
print("Saved → models/mnist_cnn.keras")

# Plot training results
plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(131)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(132)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show sample predictions
predictions = model.predict(test_images[:12], verbose=0)
pred_labels = np.argmax(predictions, axis=1)

for i in range(12):
    plt.subplot(4, 6, i + 13)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    color = 'green' if pred_labels[i] == test_labels[i] else 'red'
    plt.title(f'T:{test_labels[i]} P:{pred_labels[i]}', color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('models/training_results.png')
plt.show()

# Visualizations
plt.figure(figsize=(15, 10))

# Plot training history
plt.subplot(131)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Show sample predictions
predictions = model.predict(test_images[:12], verbose=0)
pred_labels = np.argmax(predictions, axis=1)

plt.subplot(132)
for i in range(12):
    plt.subplot(4, 6, i + 7)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    color = 'green' if pred_labels[i] == test_labels[i] else 'red'
    plt.title(f'P:{pred_labels[i]}', color=color)
    plt.axis('off')

# Show misclassified examples
y_pred = model.predict(test_images, verbose=0).argmax(axis=1)
mis_idx = np.where(y_pred != test_labels)[0][:12]

for i, idx in enumerate(mis_idx):
    plt.subplot(4, 6, i + 13)
    plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
    plt.title(f'T:{test_labels[idx]} P:{y_pred[idx]}', color='red')
    plt.axis('off')

plt.tight_layout()
plt.show()