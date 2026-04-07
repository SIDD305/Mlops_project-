import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

from dataloader import get_test   


# Load trained model
model = load_model("models/ecg_multiclass_cnn_model.keras")


# Load test dataset
test_data = get_test()


class_names = test_data.class_names
print("Classes:", class_names)
print()

correct = 0
total = 0


for images, labels in test_data:

    predictions = model.predict(images)

    predicted_class = np.argmax(predictions)
    actual_class = np.argmax(labels.numpy())

    predicted_label = class_names[predicted_class]
    actual_label = class_names[actual_class]

    print(f"Actual: {actual_label} | Predicted: {predicted_label}")

    if predicted_class == actual_class:
        correct += 1

    total += 1


accuracy = correct / total

print()
print(f"Test Accuracy: {accuracy*100:.2f}% ({correct}/{total})")