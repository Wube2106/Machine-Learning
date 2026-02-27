from network2 import Network
from mnist_loader import load_data_wrapper
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
training_data, validation_data, test_data = load_data_wrapper()

net=Network([784,30,10])
training_accuracy,evaluation_accuracy,training_loss, evaluation_loss = net.SGD(
    training_data, # Limit training data to 1000 samples for faster testing
    epochs=10,
    mini_batch_size=20,
    eta=3.0,
    lmbda=5.0,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_accuracy=True,
    monitor_training_loss=True,
    monitor_evaluation_loss=True
)

predictions = []
true_labels = []
for x, y in test_data:
    output = net.feed_forward(x)
    prediction = np.argmax(output)   # predicted digit
    true_label = y                   # actual digit

    predictions.append(prediction)
    true_labels.append(true_label)

print("Final test accuracy:", evaluation_accuracy[-1], "/", len(test_data))
print("Final test accuracy in %:", (evaluation_accuracy[-1]/len(test_data)*100))

cm = confusion_matrix(true_labels, predictions)
print(cm)

# Loss plot
plt.plot(training_loss, label="Training Loss")
plt.plot(evaluation_loss, label="Evaluation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy plot
plt.plot(training_accuracy, label="Training Accuracy")
plt.plot(evaluation_accuracy, label="Evaluation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()





