import torch
import glob
import os
import open3d as o3d
import numpy as np
from torch_geometric.data import  InMemoryDataset, Data
from torch_geometric.loader import  DataLoader
import torch_geometric.transforms as T
import json


class EvitadoDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        train = True,
        split_file = 'split.json',
        transform= None,
        pre_transform= None,
        pre_filter= None
    ):
        self.split_file = split_file
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        # self.data = torch.load(path)


    @property
    def raw_file_names(self):
        # If the strings are not present in the raw data folder  it looks to download
        return [
                "737_500",
                "737_800",
                "737_max8",
                "747_400",
                "757_200",
                "767_300",
                "777_200",
                "777_300",
                "777_f",
                "a300_600",
                "a319_ceo",
                "a320",
                "a321",
                "a330_200",
                "a330_300",
                "a340_600",
                "a350_900",
                "a380_800",
                "ch_47",
                "citation",
                "e190",
                "f16",
                "falcon_900",
                "global",
                "gulfstream",
                "legacy500",
                "md_11",
                "nh90",
                "pa34_seneca",
                "phenom300",
                "piaggio_p180",
                "pilatus_pc12",
                "premier_1",
                "tbm"
                ]

    @property
    def processed_file_names(self):
        return ['train.pt','test.pt']

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])
    
    def process_set(self, dataset):
        # categories = glob.glob(os.path.join(self.raw_dir, '*', ''))
        # categories = sorted([x.split(os.sep)[-2] for x in categories])
        categories = self.get_categories(dataset)
        
        data_list = []
        path_list= []
        for target, category in enumerate(categories):
            print(target, category)
            folder = os.path.join(self.raw_dir,category,dataset)
            paths = glob.glob(f'{folder}/{category}*.p*')
            for path in paths:                                       
                pcd = o3d.io.read_point_cloud(path)
                pos = torch.from_numpy(np.asarray(pcd.points)) 
                # print((pos.dtype))
                # data = read_ply(path)
                data = Data(pos=pos.double(), face=None)
                data.y = torch.tensor([target])
                data_list.append(data)
                path_list.append(path)
                
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
         
        return self.collate(data_list)
    
    def get_categories(self, split):
        f = open(self.raw_dir + '/' + self.split_file)
        print(self.raw_dir + '/' + self.split_file)
        data = json.load(f)
        f.close()
        return data[split]

    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'


                 




def visualize_point_cloud(points, color='r'):
    '''
    points: (N, 3)
    color: string, ['r', 'g']
    '''
    colors = np.zeros(points.shape, dtype=np.double)
    if color=='r':
        colors[:, 0] = 1    # red
    elif color=='g':
        colors[:, 1] = 1    # green
    elif color=='b':
        colors[:, 2] = 1    # blue

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


# Run the below code to visualise the data

# train_dataset = EvitadoDataset('../data_root/evitado_data_full', train=True, pre_transform=T.NormalizeScale(),transform=T.FixedPoints(1024), split_file='split.json')

# for i in train_dataset:
#     pcl = visualize_point_cloud(i.pos.numpy())
#     o3d.visualization.draw_geometries([pcl])
    
# test_dataset = EvitadoDataset('../data_root/evitado_data_full', train=False,
                                #  pre_transform=T.NormalizeScale(), transform=T.FixedPoints(1024), split_file='split.json')
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=1, drop_last=True)

# from model_utils import simulate_partial_point_clouds

# for j, data in enumerate(test_dataloader, 0):
#         # data = data.to(device)
#         data_observed = simulate_partial_point_clouds(data, 512, "classification")
#         pos_observed, batch_observed, label_observed = data_observed.pos, data_observed.batch, data_observed.y
#         pcl = visualize_point_cloud(pos_observed.numpy())
#         o3d.visualization.draw_geometries([pcl])