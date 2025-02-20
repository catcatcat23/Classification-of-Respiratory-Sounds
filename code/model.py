import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4))  # Adjust pooling layer parameters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))  # Adjust pooling layer parameters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 4))  # Adjust pooling layer parameters

        # Dynamically compute LSTM input size
        self.lstm_input_size = 8 # Adjust according to convolutional output size
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), x.size(1), -1)  # Flatten to (batch_size, sequence_length, features)
        
        x, _ = self.lstm(x)  # Pass through LSTM
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # Use last time-step's output
        return x
