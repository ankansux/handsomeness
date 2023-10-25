import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Data Preprocessing
train_dataset = image_dataset_from_directory(
    'me', validation_split=0.2, subset="training", seed=123, image_size=(150, 150), batch_size=32)
val_dataset = image_dataset_from_directory(
    'me2', validation_split=0.2, subset="validation", seed=123, image_size=(150, 150), batch_size=32)

# Model Architecture
model = models.Sequential([
    layers.InputLayer(input_shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Save the model for deployment
model.save('face_recognition_model.h5')
