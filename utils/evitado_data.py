import torch
import glob
import os
import open3d as o3d
import numpy as np
from torch_geometric.data import Dataset, download_url


class EvitadoDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['737_500','737_800',
                '747_400','757_200',
                '767_300','777_200',
                '777_300','a330_300']

    @property
    def processed_file_names(self):
        return ['train.pt,test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])
    
    def process_set(self, dataset):
        categories = glob.glob(os.path.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])
        
        data_list = []
        for target, category in enumerate(categories):
            folder = os.path.join(self.raw_dir,category,dataset)
            paths = glob.glob(f'{folder}/{category}*.pcd')
            for path in paths:
                pcd = o3d.io.read_point_cloud(path)
                data = torch.from_numpy(np.asarray(pcd.points))
                data.y = torch.tensor([target])
                data_list.append(data)
                
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
            
        return self.collate(data_list)

        
EvitadoDataset(root='../data_root/evitado_data')