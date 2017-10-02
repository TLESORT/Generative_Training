class Classifier(nn.Module):
    def __init__(self):
        super(Classif, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 10)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def classif(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return F.log_softmax(self.fc3(h2))


    def forward(self, x):
        return self.classif(x.view(-1, 784))

    def train():
        print("Classic Training")

    def train_with_generator():
        
        print("A generator train me")
