import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate, activation_type, use_bias):
        """
       

        input_size   : Number of input features 
        hidden_layers: List of neuron counts for each hidden layer (e.g. [8] or [8, 4])
        output_size  : Number of output classes (e.g. 3 for species)
        """
        
        self.layers_config = [input_size] + hidden_layers + [output_size]
        self.lr = learning_rate
        self.activation_type = activation_type
        self.use_bias = use_bias
        self.weights = []
        
        # We iterate through the layers to create weight matrices between them.
        for i in range(len(self.layers_config) - 1):
            n_in = self.layers_config[i]
            n_out = self.layers_config[i+1]
            
            # Weight matrix size: (Number of inputs + 1 for bias) x (Number of outputs)
            # We initialize weights with small random uniform values between -0.5 and 0.5.
            if self.use_bias:
                shape = (n_in + 1, n_out)
            else:
                shape = (n_in, n_out)
                
            weight_matrix = np.random.uniform(-0.5, 0.5, shape)
            self.weights.append(weight_matrix)   # Store the weight matrix between layer i and i+1

           
            print(f"Initialized weight matrix between layer {i} and {i+1} with shape: {weight_matrix.shape}")

    def activation(self, x):
        
        if self.activation_type == 'Sigmoid':
            # Range: (0, 1)
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'Tanh':
            # Range: (-1, 1)
            return np.tanh(x)
        return x
    def activation_derivative(self, a):
       
        # a is the f(x) output of the activation function, so we can compute the derivative based on that.
        if self.activation_type == 'Sigmoid':
            # f'(x) = f(x) * (1 - f(x))
            return a * (1 - a)
        elif self.activation_type == 'Tanh':
            # f'(x) = 1 - f(x)^2
            return 1 - a**2
        
    
    def forward_propagation(self, X_input):
        
        layer_inputs = []
        current_a = X_input   
        
        for i in range(len(self.weights)):
            # 1. Prepare current input: Add bias term if enabled
            if self.use_bias:
                # np.hstack adds a column of 1s to the end of the feature vector
                inp = np.hstack([current_a, np.ones((current_a.shape[0], 1))])
            else:
                inp = current_a
                
            layer_inputs.append(inp) # Store for backprop later
            
            # 2. Compute Net (z) = Input * Weights
            z = np.dot(inp, self.weights[i])
            
            # 3. Compute Activation (a) = f(z)
            current_a = self.activation(z)
        
         
            print(f"Layer {i} - Net input (z):\n{z}\n")
            print(f"Layer {i} - Activation output (a):\n{current_a}\n") 




       
        return layer_inputs, current_a
    
    def backpropagation(self, layer_inputs, final_output, y_true):
        """
        1-do the backward pass 
        2- update all weights.

        Parameters:
            layer_inputs : list of arrays from forward_propagation()
                        each entry is the input to that layer (with bias col if enabled)
            final_output : the network's output after the last activation (shape: [batch, output_size])
            y_true       : ground-truth one-hot labels (shape: [batch, output_size])
        """


        error = y_true - final_output                          
        delta = error * self.activation_derivative(final_output)  

        for i in reversed(range(len(self.weights))):

            inp = layer_inputs[i]          
            W   = self.weights[i]         
            grad_W = np.dot(inp.T, delta) / inp.shape[0]      
            self.weights[i] += self.lr * grad_W                

        
            if i > 0:
                # Remove the bias row from W before propagating
                W_no_bias = W[:-1, :] if self.use_bias else W  

                # delta for layer i-1:  (delta · W.T) * f'(a_{i-1})
                # layer_inputs[i] without the appended bias column = activation of layer i-1
                prev_a = inp[:, :-1] if self.use_bias else inp  
                delta  = np.dot(delta, W_no_bias.T) * self.activation_derivative(prev_a)

        print("Weights updated successfully after backpropagation.") 

    def train(self, x_train, y_train, epochs):

        for epoch in range(epochs):
            layer_inputs, final_output = self.forward_propagation(x_train)
            self.backpropagation(layer_inputs, final_output, y_train)
            print(f"Epoch {epoch+1}/{epochs} completed.\n")
        return final_output    

    def test(self, x_test):
        _, final_output = self.forward_propagation(x_test)
        predictions =[]
        # check each row and get index of max value 
        for rows in final_output:
            row_l=list(rows)
            max_value=max(row_l)
            max_index=row_l.index(max_value)
            predictions.append(max_index)
        return np.array(predictions)
        

