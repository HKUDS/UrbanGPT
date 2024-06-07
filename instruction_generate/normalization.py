import numpy as np
import torch


class NScaler(object):
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

class ColumnMinMaxScaler():
    #Note: to use this scale, must init the min and max with column min and column max
    def __init__(self, min, max):
        self.min = min
        self.min_max = max - self.min
        self.min_max[self.min_max==0] = 1
    def transform(self, data):
        print(data.shape, self.min_max.shape)
        return (data - self.min) / self.min_max

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min_max = torch.from_numpy(self.min_max).to(data.device).type(torch.float32)
            self.min = torch.from_numpy(self.min).to(data.device).type(torch.float32)
        #print(data.dtype, self.min_max.dtype, self.min.dtype)
        return (data * self.min_max + self.min)

def one_hot_by_column(data):
    #data is a 2D numpy array
    len = data.shape[0]
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        #print(len, max, min)
        zero_matrix = np.zeros((len, max-min+1))
        zero_matrix[np.arange(len), column-min] = 1
        if i == 0:
            encoded = zero_matrix
        else:
            encoded = np.hstack((encoded, zero_matrix))
    return encoded


def minmax_by_column(data):
    # data is a 2D numpy array
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        column = (column - min) / (max - min)
        column = column[:, np.newaxis]
        if i == 0:
            _normalized = column
        else:
            _normalized = np.hstack((_normalized, column))
    return _normalized


if __name__ == '__main__':


    test_data = np.array([[0,0,0, 1], [0, 1, 3, 2], [0, 2, 1, 3]])
    print(test_data)
    minimum = test_data.min(axis=1)
    print(minimum, minimum.shape, test_data.shape)
    maximum = test_data.max(axis=1)
    print(maximum)
    print(test_data-minimum)
    test_data = (test_data-minimum) / (maximum-minimum)
    print(test_data)
    print(0 == 0)
    print(0.00 == 0)
    print(0 == 0.00)
    #print(one_hot_by_column(test_data))
    #print(minmax_by_column(test_data))