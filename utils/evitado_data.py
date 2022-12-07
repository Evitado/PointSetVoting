import torch
import glob
import os
import open3d as o3d
import numpy as np
from torch_geometric.data import  InMemoryDataset, Data
from torch_geometric.loader import  DataLoader
from torch_geometric.io import read_ply
import torch_geometric.transforms as T


class EvitadoDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        train = True,
        transform= None,
        pre_transform= None,
        pre_filter= None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        # self.data = torch.load(path)


    @property
    def raw_file_names(self):
        return [
                '737_500',
                '737_800',
                '747_400',
                '757_200',
                '767_300',
                '777_200',
                '777_300',
                'a330_300']

    @property
    def processed_file_names(self):
        return ['train.pt','test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])
    
    def process_set(self, dataset):
        categories = glob.glob(os.path.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])
        
        data_list = []
        for target, category in enumerate(categories):
            print(target, category)
            folder = os.path.join(self.raw_dir,category,dataset)
            paths = glob.glob(f'{folder}/{category}*.ply')
            for path in paths:                                       
                pcd = o3d.io.read_point_cloud(path)
                pos = torch.from_numpy(np.asarray(pcd.points))
                # data = read_ply(path)
                data = Data(pos=pos, face=None)
                data.y = torch.tensor([target])
                data_list.append(data)
                
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            import ipdb; ipdb.set_trace()
            data_list = [self.pre_transform(d) for d in data_list]
            
        return self.collate(data_list)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'

        
# train_dataset = EvitadoDataset('../data_root/evitado_data', train=True, pre_transform=T.NormalizeScale(),transform=T.FixedPoints(1024))

# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
        