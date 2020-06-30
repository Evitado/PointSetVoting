import os
import os.path as osp
import shutil
import json
import h5py
import torch
import glob
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array


class completion3D_class(InMemoryDataset):
    """The Completion3D benchmark is a platform for evaluating state-of-the-art 3D 
    Object Point Cloud Completion methods. Participants are given a partial 3D object 
    point cloud and tasked to infer a complete 3D point cloud for the object.
    <https://completion3d.stanford.edu/> 
    Completion3D dataset contains 3D shape point clouds of 8 shape categories.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"plane"`, :obj:`"cabinet"`,
            :obj:`"car"`, :obj:`"chair"`, :obj:`"lamp"`, :obj:`"couch"`,
            :obj:`"table"`, :obj:`"watercraft"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        include_normals (bool, optional): If set to :obj:`False`, will not
            include normal vectors as input features. (default: :obj:`True`)
        split (string, optional): 
            If :obj:`"test"`, loads the testing dataset.
            (default: :obj:`"test"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('http://download.cs.stanford.edu/downloads/completion3d/'
            'dataset2019.zip')

    category_ids = {
        'plane': '02691156',
        'cabinet': '02933112',
        'car': '02958343',
        'chair': '03001627',
        'lamp': '03636649',
        'couch': '04256520',
        'table': '04379243',
        'watercraft': '04530566',
    }

    def __init__(self, root, categories=None, include_normals=True,
                 split='test', transform=None, pre_transform=None,
                 pre_filter=None):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)

        self.categories = categories
        self.split = split
        super(completion3D_class, self).__init__(root, transform, pre_transform,
                                       pre_filter)

        if split == 'test':
            path = self.processed_paths[0]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self.data, self.slices = torch.load(path)
        self.data.x = self.data.x if include_normals else None

    @property
    # all the folder names except xxx.txt
    def raw_file_names(self):
        # return list(self.category_ids.values()) + ['train_test_split']
        return list(['test'])

    @property
    # naming the pt files, eg : cha_air_car_test.pt, cha_air_car_train.pt
    def processed_file_names(self):
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            os.path.join('{}_{}.pt'.format(cats, split))
            for split in ['test']
        ]

    def process_filenames(self, filenames, split_in_loop):
        data_list = []

        for name in filenames:
            #convert name (an item in the filenames list) to str
            name = str(name)

            fpos = None
            pos = None

            if split_in_loop == 'test':
                fpos = h5py.File(name, 'r')
                pos = torch.tensor(fpos['data'], dtype=torch.float32)

            result_name = os.path.splitext(os.path.basename(name))[0]
            
            data = Data(pos = pos, resName = result_name)
  
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        trainval = []
        print('in the process')
        path = glob.glob(f'{self.raw_dir}/*.h5') 
        print('path')
        print(path)
        data_list = self.process_filenames(path, 'test')
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('end of process()')

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)
