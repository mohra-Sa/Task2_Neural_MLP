# Main script to run the preprocessing and mlp files

from preprocessing import preprocess_data
from mlp import MultiLayerPerceptron



if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler, species_list = preprocess_data()
    input_size = X_train.shape[1]  # Number of features
    hidden_layers = [2,2]  # Example hidden layer configuration
    output_size = y_train.shape[1]  # Number of classes (one-hot encoded
    learning_rate = 0.01
    activation_type = 'Tanh'  # Example activation function
    use_bias = True
 



    mlp = MultiLayerPerceptron(input_size, hidden_layers, output_size, learning_rate, activation_type, use_bias)
    print("Initial Weights:")
    for i, weight_matrix in enumerate(mlp.weights):
        print(f"Weight matrix between layer {i} and {i+1}:\n{weight_matrix}\n")
 
    layer_inputs, final_output = mlp.forward_propagation(X_train)
    print("Final output of forward propagation:\n", final_output)

  






