import pickle
import gzip
import numpy as np

def load_data():
  f = gzip.open('../data/mnist.pkl.gz', 'rb')
  training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
  f.close()
  return (training_data,validation_data,test_data)

def load_data_wrapper():
  tr_d,va_d,te_d=load_data()
  training_input=[np.reshape(x,(784,1)) for x in tr_d[0]]
  training_results=[vectorized_result(y) for y in tr_d[1]]
  trainin_data=zip(training_input,training_results)
  validation_input=[np.reshape(x,(784,1)) for x in va_d[0]]
  validation_data=zip(validation_input,va_d[1])
  test_input=[np.reshape(x,(784,1))for x in te_d[0]]
  test_data=zip(test_input,te_d[1])
  return (trainin_data,validation_data,test_data)

def vectorized_result(j):
  e=np.zeros((10,1))
  e[j]=1.0
  return e

