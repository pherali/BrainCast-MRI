import numpy as np

def basic_NN(X, y, learning_rate=0.1, epochs=10):
    """
    A basic single-layer perceptron (binary output).
    
    Parameters:

    1. X - np.array
        Input feature matrix (each row = training example, each column = input feature)
    2. y - np.array
        Binary target labels (0 or 1)
    3. learning_rate - float
        Step size for weight updates
    4. epochs - int
        Number of times to loop over the entire dataset during training
    
    So y is what we want to achieve, X is what we have as input data.

    Returns:

    1. weights - np.array
        Learned weight vector
    2. bias - float
        Learned bias term

    """

    #Initialise weights (set to zeros initially)
    #shape[1] gives number of input features (2 inpput features for logic gates)
    weights = np.zeros(X.shape[1])

    #Initialise bias (threshold)
    bias = 0.0

    # Define the activation function (step function)
    #If z >= 0 → output 1; otherwise output 0
    def activation(z):
        return np.where(z >= 0, 1, 0)

    #TRAINING LOOP
    for epoch in range(epochs):
        #Loop over each training sample
        for xi, target in zip(X, y):

            #Compute the linear combination: z = w⋅x + b
            z = np.dot(xi, weights) + bias

            #Apply the activation function
            y_pred = activation(z)

            #Compute error (difference between target and prediction)
            error = target - y_pred

            #Update weights using the perceptron learning rule:
            #Δw = η * error * x
            weights += learning_rate * error * xi

            #Update bias:
            #Δb = η * error
            bias += learning_rate * error

        # Optionally, print progress per epoch
        # print(f"Epoch {epoch+1}/{epochs}, Weights: {weights}, Bias: {bias}")

    # Return learned parameters
    return weights, bias

    #In the above training loop, we loop over each training example,
    #compute the prediction, calculate the error, and update the weights and bias accordingly.
    #This process is repeated for the specified number of epochs to allow the model to learn from the data.
    # After training, the function returns the learned weights and bias.


#PREDICTION FUNCTION

def predict(X, weights, bias):
    """
    Make binary predictions using TRAINED weights and bias.
    """
    z = np.dot(X, weights) + bias          #Linear combination
    return np.where(z >= 0, 1, 0)          #Step activation



#LOGIC GATE TRAINING DATA


#Input combinations for 2-bit logic gates

# X1  X2
# 0   0
# 0   1
# 1   0
# 1   1

X = np.array([[0,0], [0,1], [1,0], [1,1]])
#X is the input matrix representing all possible combinations of two binary inputs.
#Each row corresponds to a unique input pair (X1, X2), each row is one training sample.

#Target outputs for each gate
#basically the type of outputs that we want to see from the perceptron after training.
Y_AND  = np.array([0, 0, 0, 1])
Y_OR   = np.array([0, 1, 1, 1])
Y_NAND = np.array([1, 1, 1, 0])



#TRAIN SINGLE-LAYER PERCEPTRONS FOR EACH GATE


#Train the perceptron for AND gate
w_and, b_and = basic_NN(X, Y_AND, learning_rate=0.1, epochs=20)

#Train the perceptron for OR gate
w_or, b_or = basic_NN(X, Y_OR, learning_rate=0.1, epochs=20)

#Train the perceptron for NAND gate
w_nand, b_nand = basic_NN(X, Y_NAND, learning_rate=0.1, epochs=20)



#EVALUATE TRAINED GATES


for i in X:
    print(f"Input: {i[0]} {i[1]}")

print("AND Gate Predictions:  ", predict(X, w_and, b_and))
print("OR  Gate Predictions:  ", predict(X, w_or, b_or))
print("NAND Gate Predictions: ", predict(X, w_nand, b_nand))



#MULTI-LAYER PERCEPTRON (for XOR)

#XOR cannot be solved by a single perceptron because it is not
#linearly separable — need at least one hidden layer.

#Can build XOR using outputs of OR and NAND gates as input to an AND gate.
#Logic -> XOR(x1, x2) = AND( OR(x1, x2), NAND(x1, x2) )

def XOR_gate(X):
    """
    XOR logic using trained perceptrons for OR, NAND, and AND.
    Effectively a 2-layer neural network.
    """

    #Hidden layer outputs
    or_out   = predict(X, w_or, b_or)      #First neuron: OR gate
    nand_out = predict(X, w_nand, b_nand)  #Second neuron: NAND gate

    #Combine outputs of hidden neurons as new input
    X_hidden = np.column_stack((or_out, nand_out))

    #Output neuron (AND gate on hidden layer outputs)
    xor_out = predict(X_hidden, w_and, b_and)
    return xor_out

print("XOR Gate Predictions:  ", XOR_gate(X))

