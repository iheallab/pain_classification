import numpy as np 
import torch

from torch.utils import data
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import random

def padding(h):
    sizes = np.arange(20)
    ideal_size = 0
    for x in sizes:
        if (2 ** x) >= h:
            ideal_size = 2 ** x
            break
    pad = (ideal_size - h) / 2
    return int(pad)

# Data augmentation for time-series data
# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
# T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220.

# https://dl.acm.org/citation.cfm?id=3136817
#
# https://arxiv.org/abs/1706.00527

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis, angle))

## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum


def RandSampleTimesteps(X, nSample=1000):
    X_new = np.zeros(X.shape)
    tt = np.zeros((nSample,X.shape[1]), dtype=int)
    tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,2] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[-1,:] = X.shape[0]-1
    return tt

def DA_RandSampling(X, nSample=1000):
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
    X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
    X_new[:,2] = np.interp(np.arange(X.shape[0]), tt[:,2], X[tt[:,2],2])
    return X_new

# ## 4. Time Warping

# #### Hyperparameters :  sigma = STD of the random knots for generating curves
#
# #### knot = # of knots for the random curves (complexity of the curves)

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)

def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

# Meta-dataset.
class MetaDataset(data.Dataset):

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.labels = np.concatenate([dataset['X_train'], dataset['X_val']],
                                                axis=0), np.concatenate(
            [dataset['y_train'], dataset['y_val']], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()

        sample = self.data[idx].squeeze()
        sample = sample.astype(np.float32)

        pad = padding(sample.shape[-1])
        sample = np.pad(sample, ((0, 0), (pad, pad)), mode='constant')

        target = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return (sample, target)


# Target dataset.
class TargetDataset(data.Dataset):

    def __init__(self, mode, dataset, fold, num_shots):

        self.mode = mode
        self.fold = fold
        self.dataset = dataset
        self.num_shots = num_shots

        # Setting datasets.
        self.make_dataset()

    def make_dataset(self):
        # Making sure the mode is correct.
        assert self.mode in ['train', 'tuning', 'test']
        self.items = []

        self.fewshot_tasks = np.unique(self.dataset['y_test'])

        shot = f'kfold_{self.num_shots}_shot'
        if self.mode == 'train' or self.mode == 'tuning':
            mode_str = 'train_idx'
        elif self.mode == 'test':
            mode_str = 'test_idx'

        # get idx based on num_shots, k-fold and mode
        nshot_idx = self.dataset[shot][self.fold][mode_str]

        # for each task
        for i, t in enumerate(self.fewshot_tasks):
            # get num_shot samples from that task
            for sample, label in zip(self.dataset['X_test'][nshot_idx], self.dataset['y_test'][nshot_idx]):
                if t == label:
                    item = {'acc': sample,
                            'label': i}
                    self.items.append(item)

    def __getitem__(self, index):

        # Reading items from list.
        data = self.items[index]

        acc = np.squeeze(data['acc'])
        lab = data['label']

        # Casting image to the appropriate dtype.
        acc = acc.astype(np.float32)

        # pad goes here
        self.pad = padding(acc.shape[-1])

        acc = np.pad(acc, ((0, 0), (self.pad, self.pad)), mode='constant')

        # Turning to tensors.
        acc = torch.from_numpy(acc)

        return acc, lab

    def __len__(self):

        return len(self.items)


class SensorSequence(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.classes = np.unique(labels)
        self.class_to_idx = {v: k for k, v in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx].squeeze()
        sample = np.transpose(sample, (1, 0))
        # split accelerometer into chunks of size 1 minute (600 values = 60 seconds x 10hz)
        n = 600
        chunks = [sample[:, i:i + n] for i in range(0, sample.shape[1], n)]
        sequences = []
        for chunk in chunks:
            chunk = chunk.astype(np.float32)
            padding_size = padding(chunk.shape[-1])
            chunk = np.pad(chunk, ((0, 0), (padding_size, padding_size)), mode='constant')
            sequences.append(chunk)
        target = self.class_to_idx[self.labels[idx]]

        return np.array(sequences), [], target


class SensorDataset(data.Dataset):
    def __init__(self, data, labels, sample_start, dataaug=False):
        self.data = data["acc"]
        self.clin_data = data["clin"]
        self.labels = labels
        self.dataaug = dataaug
        #self.padding_size = padding(self.data[1].shape[-1])
        self.classes = np.unique(labels)
        self.sample_start = sample_start

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx].squeeze()
        clin_data = self.clin_data[idx]
        sample = sample[self.sample_start:]

        if self.dataaug:
            if np.random.rand() > 0.5:
                sample = DA_Rotation(sample)

        # normalizacao
        sample = ((sample - np.min(sample)) / (np.max(sample) - np.min(sample))) * 2 - 1
        sample = (sample - sample.mean()) / sample.std()

        sample = np.transpose(sample, (1, 0))

        # padding_size = padding(sample.shape[-1])
        # sample = np.pad(sample, ((0, 0), (padding_size, padding_size)), mode='constant')

        sample = sample.astype(np.float32)
        #sample = np.pad(sample, ((0, 0), (self.padding_size, self.padding_size)), mode='constant')
        target = self.labels[idx]

        if np.isnan(sample).any():
            raise ValueError('Nan values in sample after')
        data = {"acc": sample, "clin": np.array(clin_data, dtype=np.float32)}
        return data, target


class SensorPublicDataset(data.Dataset):
    def __init__(self, data, add_data, labels, transform=None, target_transform=None):
        self.data = data
        self.add_data = add_data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.padding_size = padding(self.data[0].shape[0])
        self.classes = np.unique(labels)
        self.class_to_idx = {v: k for k, v in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx].squeeze()
        add_data_sample = []
        if self.add_data is not None:
            add_data_sample = self.add_data[idx]

        #sample = np.transpose(sample, (1, 0))
        # if random.randrange(3) == 0:
        #     sample = DA_Rotation(sample)
        # if random.randrange(3) == 0:
        #     sample = DA_Permutation(sample, minSegLength=20)
        sample = np.transpose(sample, (1, 0))

        sample = sample.astype(np.float32)
        sample = np.pad(sample, ((0, 0), (self.padding_size, self.padding_size)), mode='constant')
        target = self.class_to_idx[self.labels[idx]]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(sample)

        return (sample, add_data_sample, target)


if __name__ == '__main__':
    fold = 0
    train_shots = 5
    test_shots = 5
    ways = 4
    meta_bsz = 3

    file_path = 'Y:/sensors/nshot_experiments/pra_rodar/f20_t5_4ways_target[pamap2]_source[_mhealth_wharf_wisdm_uschad]_exp1_d_FSL.npz'
    dataset = np.load(file_path, allow_pickle=True)

    # Datasets and dataloaders. dataset, fold, num_shots, num_ways, inner_iters)
    metadataset = MetaDataset(dataset, fold, train_shots, ways, meta_bsz)

    tune_dataset = TargetDataset('tuning', dataset, fold, test_shots)

    test_dataset = TargetDataset('test', dataset, fold, test_shots)

