import numpy as np
class Network():
  def __init__(self,sizes):
    self.sizes=sizes
    self.num_Layers=len(sizes)
    self.biases=[np.random.randn(y,1) for y in sizes[1:]]
    self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
   
  def sigmoid(self,z):
    return 1.0/(1.0+np.exp(-z))
  
  def feedforward(self, a):
    for w,b in zip(self.weights,self.biases):
        a=self.sigmoid(np.dot(w,a)+b)
    return a
  
  def SGD(self, training_data, epochs,mini_batch_size,eta,test_data=None):
    if test_data:n_test=len(test_data)
    n = len(training_data)
    for x in np.xrange(epochs):
        np.random.shuffle(training_data)
        mini_batches=[training_data[k:k+mini_batch_size]
        for k in range(n,mini_batch_size)]

        for mini_batch in mini_batches:
            self.updated_mini_batch(mini_batch,eta)

  def updated_mini_batch(self,mini_batch,eta):
    nabla_b=[np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    for x,y in mini_batch:
        delta_nabla_b,delta_nabla_w=self.propagation(x,y)
        
        nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
        nabla_w=[nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights=[w-eta/(len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b -eta / (len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
