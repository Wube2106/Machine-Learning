import numpy as np
from mnist_loader import vectorized_result
class cross_entropy_fun():
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y):
        return a-y

class quadratic_cost_fun():
    @staticmethod
    def fn(a,y):
        #return 0.5*np.sum(np.square(y-a)) both are the same
        return 0.5*np.linalg.norm(y-a)**2

    
    @staticmethod
    def delta(z,a,y):
        sigmoid = 1.0/(1.0+np.exp(-z))
        return (a-y)*sigmoid*(1-sigmoid)


class Network():
    def __init__(self,sizes,cost=cross_entropy_fun): # type: ignore
        self.sizes=sizes
        self.num_layers=len(sizes)
        self.cost=cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        self.biases=[np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def large_weight_initializer(self):#the older one which we used in network.py
        self.biases=[np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def feed_forward(self,a):
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,a)+b
            a=self.sigmoid(z)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,lmbda=0.0,evaluation_data=None,
            monitor_evaluation_accuracy=False,
            monitor_training_accuracy=False,
            monitor_training_loss=False,
            monitor_evaluation_loss=False
            ):
        n = len(training_data)

        training_accuracy,evaluation_accuracy=[],[]
        training_loss,evaluation_loss=[],[]

        for x in range(epochs):
            np.random.shuffle(training_data)
            mini_batchs=[training_data[k:k+mini_batch_size]
                         for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta,lmbda,n)

            print(f"Epoch {x+1}")
            if monitor_training_loss:
                loss=self.total_cost(training_data,lmbda,convert=False)
                training_loss.append(loss)
                print(f"Training loss: {loss}")
            if monitor_training_accuracy:
                accuracy=self.accuracy(training_data,convert=True)/len(training_data)
                training_accuracy.append(accuracy)
                print(f"Training accuracy: {accuracy}")
            if monitor_evaluation_loss:
                loss=self.total_cost(evaluation_data,lmbda,convert=True)
                evaluation_loss.append(loss)
                print(f"Evaluation loss: {loss}")
            if monitor_evaluation_accuracy:
                accuracy=self.accuracy(evaluation_data,convert=False)/len(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Evaluation accuracy: {accuracy}")
        return training_accuracy,evaluation_accuracy, training_loss, evaluation_loss
    
    def update_mini_batch(self,mini_batch,eta,lmbda,n):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.back_propagation(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights=[w*(1-eta*(lmbda/n))-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    def back_propagation(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation=x
        activations=[x]
        zs=[]

        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=self.sigmoid(z)
            activations.append(activation)

        delta=(self.cost).delta(zs[-1],activations[-1],y)
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        for l in range(2,self.num_layers):
            z= zs[-l]
            sp=self.sigmoid_prime(z)

            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
            nabla_b[-l]=delta

        return (nabla_b,nabla_w)


    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def accuracy(self,data,convert=False):
        if convert:
            results=[(np.argmax(self.feed_forward(x)),np.argmax(y))
                     for (x,y) in data]
        else:
            results=[(np.argmax(self.feed_forward(x)),y)
                     for (x,y) in data]
        return sum((x==y) for (x,y) in results)
    def total_cost(self,data,lmbda, convert=False):#regularization term is added to the cost function
        cost=0.0
        for (x,y) in data:
            a=self.feed_forward(x)
            if convert:
                y=vectorized_result(y)
            cost+=self.cost.fn(a,y)/len(data)
        cost+=0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
 








