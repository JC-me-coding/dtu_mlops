from tests import _PATH_DATA
from data import mnist

def testdata():
    dataset = mnist()
    train = dataset[0]
    test = dataset[1]

    N_train = 25000 or 40000
    N_test = 5000
    N_labels = 10

    assert len(train) == N_train
    assert len(test) == N_test

'''
@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
'''