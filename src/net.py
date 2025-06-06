class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # size of the image is 16*16*128
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*16*128, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.fc(x)

        return x

model = Net(in_channels=3, num_classes=8)