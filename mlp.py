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
    