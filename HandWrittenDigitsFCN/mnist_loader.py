import gzip
import pickle
import numpy as np
import warnings

def load_data():
    """
    Load the MNIST dataset from 'data/mnist.pkl.gz'.
    Returns: (training_data, validation_data, test_data)
    """
    # Suppress all warnings in this block (including old dtype alignment warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open("data/mnist.pkl.gz", "rb") as f:
            training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data

def vectorized_result(j):
    """Return a 10-dimensional unit vector with 1.0 in the jth position."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    """
    Wraps the MNIST data into tuples (x, y) suitable for Network class.
    Training data: (784x1 input, 10x1 one-hot vector)
    Validation/Test data: (784x1 input, scalar label)
    """
    tr_d, va_d, te_d = load_data()
    
    # Training data
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    # Validation data
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    # Test data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return training_data, validation_data, test_data
