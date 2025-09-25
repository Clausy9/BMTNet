import torch.utils.data as data
import os
import numpy as np
import torch

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def read_bin_file(filepath):
    '''
        read '.bin' file to 2-d numpy array

    :param path_bin_file:
        path to '.bin' file

    :return:
        2-d image as numpy array (float32)

    '''

    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]

    data_2d = data[2:].reshape((hh, ww))
    data_2d = data_2d.astype(np.float32)

    return data_2d

class Demosaic_test(data.Dataset):
    def __init__(self, QB_folder):
        super(Demosaic_test, self).__init__()
        
        self.qb_files = []
        
        for qb_image in os.listdir(QB_folder):
            qb_image_file = os.path.join(QB_folder, qb_image)
            self.qb_files.append(qb_image_file)
        
        LEN = len(os.listdir(QB_folder))
        self.nums_trainset = LEN
        
    def __len__(self):
        return self.nums_trainset

    def __getitem__(self, idx):
        idx = idx % self.nums_trainset
        qb = read_bin_file(self.qb_files[idx])
        qbayer_sg = qb.copy()
        back = {}
        back['qbayer_sg_name'] = self.qb_files[idx]
        back['qbayer_sg'] = ndarray2tensor(qbayer_sg[..., np.newaxis])

        return back