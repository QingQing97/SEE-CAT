import torch.nn as nn

'''
3-layer MLP with ReLU activation.
'''
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, int(0.5 * input_dim))  # Changed to int
        self.relu1 = nn.ReLU()  # Changed to separate instance
        self.linear_2 = nn.Linear(int(0.5 * input_dim), hidden_dim)
        self.relu2 = nn.ReLU()  # Changed to separate instance
        self.linear_3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu1(self.linear_1(x))  # Apply ReLU after first linear layer
        x = self.dropout(x)                # Apply dropout after first ReLU
        x = self.relu2(self.linear_2(x))  # Apply ReLU after second linear layer
        x = self.dropout(x)                # Apply dropout after second ReLU
        x = self.linear_3(x)               # Final linear layer without activation
        return x

