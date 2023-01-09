from tests import _PATH_DATA
from data import mnist

dataset = mnist()
train = dataset[0]
N_train = 250000 or 40000
N_test = 5000
assert len(train) == N_train 