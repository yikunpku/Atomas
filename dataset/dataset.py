from torch.utils.data import Dataset
import os.path as osp
import csv
import pickle
import os
import torch

class ChEBI_20_data_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        ):
        self.data_path = data_path
        self.cids = []
        self.descriptions = {}
        self.smiles = {}
        
        #load data
        with open(osp.join(data_path, dataset, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'smiles', 'desc'], skipinitialspace=True)
            next(reader)
            for n, line in enumerate(reader):
                self.descriptions[line['cid']] = line['desc']
                self.smiles[line['cid']] = line['smiles']
                self.cids.append(line['cid'])

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, idx):

        cid = self.cids[idx]

        smiles = self.smiles[cid]

        description = self.descriptions[cid]


        return {
                'description':description,
                'smiles':smiles
                }        
        
class PubChem_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        ):
        self.data_path = data_path

        #load data
        with open (osp.join(data_path, dataset, split+'.pkl'),"rb") as f:
            self.data = pickle.load(f)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # cid = self.data[idx]["CID"]

        smiles = self.data[idx]["smiles"]

        description = self.data[idx]["text"]

        return {
                'description':description,
                'smiles':smiles
                }

class PCdes_CLMP_Dataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset,
        split,
        ):
        self.data_path = data_path
        self.descriptions = {}
        self.smiles = {}

        #load data
        with open(osp.join(data_path, dataset, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['smiles', 'desc'], skipinitialspace=True)
            next(reader)
            for n, line in enumerate(reader):
                self.descriptions[n] = line['desc']
                self.smiles[n] = line['smiles']
            

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):

        smiles = self.smiles[idx]
        description = self.descriptions[idx]


        return {
                'description':description,
                'smiles':smiles
                }  
    

        
    



