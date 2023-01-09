import torch
import numpy as np
from tests import _PROJECT_ROOT
from tests import _PATH_DATA
from data import mnist

#implement at least a test that checks for a given input with shape X that the output of the model have shape Y
dataset = mnist()

train_set = dataset[0]
test_set = dataset[1]

test_data = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

for data in test_data:
    image, label = data
    break
assert outputs.shape == (1,10)