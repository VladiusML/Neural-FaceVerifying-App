import torch
import torch.nn as nn

# https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
class SiameseNN(nn.Module):
    def __init__(self):
        """
        Initializes the Siamese Neural Network model architecture.
        """
        super(SiameseNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=10, stride=1),
                                    nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)  

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1),
                                    nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)  

        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
                                    nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1),
                                    nn.ReLU(inplace=True))

        self.linear = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())

        
    def forward_once(self, x):
        """
        Forward pass through one branch of the Siamese Neural Network.

        Args:
        - x: Input image tensor.

        Returns:
        - x: Output tensor after passing through the network branch.
        """
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.dropout1(x)   

        x = self.layer2(x)
        x = self.pool2(x)
        x = self.dropout2(x)  

        x = self.layer3(x)
        x = self.pool3(x)

        x = self.layer4(x)

        x = x.view(-1)
        x = self.linear(x)

        return x

    def forward(self, x1, x2):
        """
        Forward pass through the Siamese Neural Network.

        Args:
        - x1: Input image tensor for the first branch.
        - x2: Input image tensor for the second branch.

        Returns:
        - scores: Similarity scores between input images.
        """
        h1 = self.forward_once(x1)
        h2 = self.forward_once(x2)

        dist = torch.abs(h1 - h2)
        scores = self.out(dist)

        return scores