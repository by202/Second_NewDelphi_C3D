import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from utee import misc
from utee import act
print = misc.logger.info

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist.pth',
    'mnist_secure_ml': 'mnist_secure_ml.pth',
    'mnist_cryptonets': 'mnist_cryptonets.pth',
    'mnist_deepsecure': 'mnist_deepsecure.pth',
    'mnist_minionn': 'mnist_minionn.pth',
}

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()

        self.features = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MNISTCryptoNets(nn.Module):
    def __init__(self):
        super(MNISTCryptoNets, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 5, 5, 2, 2), act.Square()
        )

        self.classifier = nn.Sequential(
            nn.Linear(980, 100), act.Square(), # nn.Dropout(0.2),
            nn.Linear(100, 10)
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print("Num features", list(x.size()))
        x = self.classifier(x)
        return x

class MNISTDeepSecure(nn.Module):
    def __init__(self):
        super(MNISTDeepSecure, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 5, 5, 2, 2), nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(980, 100), nn.ReLU(), # nn.Dropout(0.2),
            nn.Linear(100, 10)
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print("Num features", list(x.size()))
        x = self.classifier(x)
        return x

class MNISTMiniONN(nn.Module):
    def __init__(self):
        super(MNISTMiniONN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5, 1),
            nn.ReLU(), nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 100), nn.ReLU(), # nn.Dropout(0.2),
            nn.Linear(100, 10)
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MNISTSecureML(nn.Module):
    def __init__(self):
        super(MNISTSecureML, self).__init__()

        self.features = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 128), act.Square(), nn.Dropout(0.2),
            nn.Linear(128, 128), act.Square(), nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs

def get(model_name, model_dir, pretrained=False):
    if model_name == 'mnist':
        model = MNIST()
    elif model_name == 'mnist_secure_ml':
        model = MNISTSecureML()
    elif model_name == 'mnist_cryptonets':
        model = MNISTCryptoNets()
    elif model_name == 'mnist_deepsecure':
        model = MNISTDeepSecure()
    elif model_name == 'mnist_minionn':
        model = MNISTMiniONN()
    elif model_name == 'C3D':
        model = C3D()
    else:
        assert False, model_name

    if pretrained:
        m = model_zoo.load_url(model_urls[model_name], model_dir)
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

