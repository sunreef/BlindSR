import torch

class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, growth_factor, layers):
        super(DenseBlock, self).__init__()
        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels + i * growth_factor, in_channels + i * growth_factor, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels + i * growth_factor, growth_factor, 3, padding=1)
            ) for i in range(layers)
        ])
        self.relu_layer = torch.nn.ReLU()
        self.final_conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels + layers * growth_factor, in_channels, 1),
            torch.nn.ReLU()
        )


    def forward(self, *input):
        current_features = input[0]
        for i in range(len(self.conv_layers)):
            conv_output = self.conv_layers[i](current_features)
            current_features = torch.cat([current_features, conv_output], 1)
            current_features = self.relu_layer(current_features)
        return self.final_conv_layer(current_features)







