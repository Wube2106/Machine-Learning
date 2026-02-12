import numpy as np
class Network():
    def __int__(self,sizes,cost=cross_entropy_fun):
        self.sizes=sizes
        self.num_layers=len(sizes)
        self.cost=cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        self.biases=[np.random.rand(y,1) for y in self.sizes[1:]]
        self.weights=[np.random.rand(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def large_weight_initializer(self):#the older one which we used in network.py
        self.biases=[np.random.rand(y,1) for y in self.sizes[1:]]
        self.weights=[np.random.rand(y,x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def feed_forward(self,a):
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,a)+b
            a=self.sigmoid(z)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,lmbda=0.0,evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
            ):
        n = len(training_data)
        if evaluation_data:eval_n=len(evaluation_data)

        evaluation_cost,evaluation_accuracy=[],[]
        training_cost,training_accuracy=[],[]

        for x in range(epochs):
            np.random.shuffle(training_data)
            mini_batchs=[training_data[k:k+mini_batch_size]
                         for k in range(n,mini_batch_size)]

            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta,lmbda,)

            if monitor_training_cost:
                cost=self.total_cost(training_data,lmbda)
                training_cost.append(cost)
            if monitor_training_accuracy:
                accuracy=self.accuracy(training_data,convert=True)
                training_accuracy.append(accuracy)
            if monitor_evaluation_cost:
                cost=self.total_cost(evaluation_data,lmbda,convert=True)
                evaluation_cost.append(cost)
            if monitor_training_accuracy:
                accuracy=self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)

        return evaluation_cost, evaluation_accuracy,training_cost, training_accuracy

    def update_mini_batch(self,mini_batch,eta,lmbda,n):
        nabla_b=[np.zeros(b.shape()) for b in self.biases]
        nabla_w = [np.zeros(w.shape()) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.back_propagation(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            self.weights=[w*(1-eta*(lmbda/n))-(eta/len(mini_batch))*dnw for w,dnw in zip(self.weights,delta_nabla_w)]
            self.biases=[b-(eta/len(mini_batch))*dnb for b,dnb in zip(self.biases,delta_nabla_b)]

    def back_propagation(self,x,y):
        nabla_b = [np.zeros(b.shape()) for b in self.biases]
        nabla_w = [np.zeros(w.shape()) for w in self.weights]

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
            z=zs[-1]
            sp=self.sigmoid_prime(z)

            delta=np.dot(self.weights[-l+1],delta)*sp
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose)
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
    def total_cost(self,data,lmbda, convert=False):
        cost=0.0
        for (x,y) in data:
            a=self.feed_forward(x)
            if convert:vectorized_result(y)
            cost+=self.cost.fn(a,y)/len(data)
        cost+=0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost


class cross_entropy_fun():
    def fn(self,a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def delta(self,z,a,y):
        return a-y

class quadratic_cost_fun():
    def fn(self,a,y):
        #return 0.5*np.sum(np.square(y-a)) both are the same
        return 0.5*np.linalg.norm(y-a)**2

    def delta(self,z,a,y):
        return (a-y)*self.sigmoid_prime(z)








