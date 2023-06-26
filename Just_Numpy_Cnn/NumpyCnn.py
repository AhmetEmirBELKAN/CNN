import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.random.randn(hidden_size)
        self.bias2 = np.random.randn(output_size)
    
    def forward_propagation(self, inputs):
        self.hidden_layer = sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
    
    def backward_propagation(self, inputs, outputs, learning_rate):
        
        output_error = outputs - self.output_layer
        output_delta = output_error * sigmoid_derivative(self.output_layer)
        
        
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)
        
        
        self.weights2 += learning_rate * np.dot(self.hidden_layer.T, output_delta)
        self.weights1 += learning_rate * np.dot(inputs.T, hidden_delta)
        
        
        self.bias2 += learning_rate * np.sum(output_delta, axis=0)
        self.bias1 += learning_rate * np.sum(hidden_delta, axis=0)
    
    def train(self, inputs, outputs, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward_propagation(inputs)
            self.backward_propagation(inputs, outputs, learning_rate)
            
            
            if epoch % 100 == 0:
                error = np.mean(np.abs(outputs - self.output_layer))
                print(f"Epoch: {epoch}, Error: {error}")
    
    def predict(self, inputs):
        self.forward_propagation(inputs)
        return self.output_layer

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

neural_net = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

neural_net.train(inputs, outputs, epochs=1000, learning_rate=0.1)

test_input = np.array([[0, 1]])
prediction = neural_net.predict(test_input)
print(f"Tahmin: {prediction}")
