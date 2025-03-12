import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """
    def __init__(self, input_size, hidden_sizes, output_size, num_layers, use_bn, lr):
        """ Initialize the model.
        """
        # Initialize the class
        super().__init__()
        
        # Set the members
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Set the sequence of sizes of layers
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        # Implement the model
        model = nn.Sequential()
        for i in range(num_layers):
            model.add_module(f"fc{i+1}", nn.Linear(sizes[i], sizes[i+1]))
            if i != num_layers - 1:
                if use_bn == True:
                    # Add batch normalization between Linear layer and ReLU activation
                    model.add_module(f"bn{i+1}", nn.BatchNorm1d(sizes[i+1]))
                model.add_module(f"relu{i+1}", nn.ReLU())
            else:
                model.add_module("sigmoid", nn.Sigmoid())
        self.model = model

        # Set the loss function and the optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)


    def forward(self, X):
        """ Forward pass.
        """
        return self.model(X)
    

    def backward_pass(self, X, y):
        """ Backward pass.
        """
        # Clean the gradient
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.forward(X)

        # Compute the loss and update the parameters
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

