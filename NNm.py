import numpy as np


class Neural_Network:

    def __init__(self):

        self.ip = 2

        self.oup =1

        self.X = np.array(([[2, 9], [1, 5], [3, 5],[3, 10],[8, 9],[12, 23], [22, 10]]), dtype=float)
        self.Y = np.array(([[22], [12], [16], [26], [35],[70], [64]]), dtype=float)

        self.b = np.array(([1]),dtype = float)
        self.xPredicted = np.array(([2,13]), dtype=float)

        self.W1 = np.array(([1],[1]),dtype=float)
        #self.W1 = self.W1/self.ip**.5

    def forward(self ,X1):

        z = np.dot(X1,self.W1)

        z= z+ self.b

        return z

    def backward(self,X1,y1,o):

        self.o_er = y1-o
        #print ("self.o_er: \n" + str(self.o_er))

        self.z2_er = np.array([X1]).T.dot(np.array([self.o_er]))
        #print ("self.z2_er: \n" + str(self.z2_er))
        #print ("Wbefore: \n" + str(self.W1))

        self.W1 += (0.001)*self.z2_er
        self.b += (0.001)*self.o_er
        #print ("b bias: \n" + str(self.b))  
        #print ("W afer: \n" + str(self.W1))

        

    def predict(self):

          print ("Predicted data based on trained weights: ")
          print ("Input (scaled): \n" + str(self.xPredicted))
          print ("Output: \n" + str(self.forward(self.xPredicted)))

    

    def saveWeights(self):

        np.savetxt("m1.txt" , self.W1,fmt = "%s")

    def train(self ,X1,Y1):

        rslt = self.forward(X1)

        self.backward(X1,Y1,rslt)

    def start(self):
        
        for j in range(1000): # trains the NN 1,000 times
              i=0
              while i <len(self.X):
                """print ("# " + str(j) + "\n")
                print ("Input (scaled): \n" + str(self.X[i]))
                print ("Actual Output: \n" + str(self.Y[i]))
                print ("Predicted Output: \n" + str(NN.forward(self.X[i])))"""
                NN.forward(self.X[i])
                self.train(self.X[i], self.Y[i])
                i+=1
 
        self.saveWeights()
        self.predict()

    
NN = Neural_Network()
NN.start()
NN.predict()



    

     

    
