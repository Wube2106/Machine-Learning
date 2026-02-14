from network2 import Network
from mnist_loader import load_data_wrapper
training_data, validation_data, test_data = load_data_wrapper()
#training_data= training_data[0:5000]
#test_data=test_data[:1000]
net=Network([784,30,10])
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
    training_data, # Limit training data to 1000 samples for faster testing
    epochs=30,
    mini_batch_size=20,
    eta=3.0,
    lmbda=5.0,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_accuracy=True,
    monitor_evaluation_cost=False,
    monitor_training_cost=False
)

print("Final test accuracy:", evaluation_accuracy[-1], "/", len(test_data))



