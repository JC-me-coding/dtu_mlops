import torch
from tests import _PROJECT_ROOT
from tests import _PATH_DATA
from model import MyAwesomeModel
from data import mnist

dataset = mnist()
testing = dataset[1]

model = MyAwesomeModel()
model.eval()
testloader = torch.utils.data.DataLoader(testing, batch_size=1,
                                         shuffle=False, num_workers=4)
model.eval()


#implement at least a test that checks for a given input with shape X that the output of the model have shape Y
dataset = mnist()

test_set = dataset[1]
test_dl = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

model = MyAwesomeModel()
model.eval()

for data in range(len(test_data)):
    image, label = data
    outputs = model(image.float())
    break
assert outputs.shape == (1,10)