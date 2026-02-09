from network import sigmoid_prime
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
        if test_data:
            print("Epoch {0}: {1} / {2}".format(x, self.evaluate(test_data), n_test))

  def updated_mini_batch(self,mini_batch,eta):
    nabla_b=[np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    for x,y in mini_batch:
        delta_nabla_b,delta_nabla_w=self.propagation(x,y)
        
        nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
        nabla_w=[nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights=[w-eta/(len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b -eta / (len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        
  def propagation(self,x,y):
     nabla_b=[np.zeros(b.shape) for b in self.biases]
     nabla_w=[np.zeros(w.shape) for w in self.weights]
     activation=x
     activations=[x]
     zs=[]

     for w, b in zip(self.weights,self.biases):
        z=np.dot(w,activations[-1])+b
        zs.append(z)
        activation=self.sigmoid(z)
        activations.append(activation)
     
     delta=self.cost_derivative(activations[-1],y)*self.sigmoid_prime(zs[-1])
     nabla_b=delta
     nabla_w=np.dot(delta,activations[-2].transpose())

     for l in range(2,self.num_layers):
        z=zs[-l+1]
        sp=sigmoid_prime(z)
        delta=np.dot(self.weights[-l],delta)*sp
        navla-b[-l]=delta
        nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
     
     return (nabla_b,nabla_w)
  
  def cost_derivative(self,output_activations,y):
     return (output_activations -y)

  def sigmoid_prime(self,x):
     return self.sigmoid(x)*(1-self.sigmoid(x))
  