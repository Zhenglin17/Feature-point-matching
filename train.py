from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from mydataset import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Two_patches(nn.Module):
    def __init__(self):
        super(Two_patches, self).__init__()
        self.all_layers = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=64, kernel_size=3, stride=2, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=3 // 2),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Linear(9, 1))

    def forward(self, x):
        x = self.all_layers(x)
        x = x.squeeze()
        m = nn.Linear(256, 1)
        m.to(device)
        x = m(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x):
        compare = 1 - x
        upper = torch.max(torch.tensor(0), compare)
        loss = torch.sum(upper)
        return loss


# event_volume = torch.randn(64, 40, 20, 20)
# Net = Two_patches()
# output = Net(event_volume)
# print(output.shape)

if __name__ == '__main__':
    myDataloader = get_dataloader()
    train_loss = 0
    model = Two_patches()
    model_loss = CustomLoss()
    model.to(device)
    model_loss.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    for epoch in range(6):
        for data_1, data_2 in myDataloader:
            # print(data_1.shape)
            data = torch.cat((data_1, data_2), dim=1)
            data = data.to(device)
            # print(data.shape)
            optimizer.zero_grad()
            output = model(data)
            loss = model_loss(output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(train_loss)

