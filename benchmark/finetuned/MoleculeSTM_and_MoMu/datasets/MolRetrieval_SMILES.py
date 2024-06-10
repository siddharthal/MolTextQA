import sys
sys.path.append('.')

import copy
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Optional
from collections.abc import Sequence

from torch.utils.data import Dataset
from torch_geometric.data.dataset import IndexType
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class Mol_retrieval_SMILES(Dataset):

    def __init__(self, root):
        
        self.root = root
        self.qa_file = os.path.join(self.root, "MolRetrieval", "raw", "data_splits.csv")

        df = pd.read_csv(self.qa_file)

        self.SMILES_list, self.sentence_list, self.options_list, self.labels_list, self.split_list, self.index_list = list(), list(), list(), list(), list(), list()
        
        for i in tqdm(range(len(df))):

            try:
            
                SMILES = df["SMILES"][i]
                
                question = df["question"][i]
                options = eval(df["options"][i])
                assert len(options) == 5
                label = int(df["correct"][i] - 1)
                sentence = question + options[label]

                smiles_options = eval(df["smiles_option"][i].replace("\n", "").replace(" ", ","))
                assert len(smiles_options) == 5
                
                label = int(df["smiles_correct"][i] - 1)
                
                split = df["split"][i]

                self.SMILES_list.append(SMILES)
                self.sentence_list.append(sentence)
                self.options_list.append(smiles_options)
                self.labels_list.append(label)
                self.split_list.append(split)
                self.index_list.append(i)
            
            except:
                pass
        
        print("Prepared data : {}".format(len(self.SMILES_list)))

        return

    def __getitem__(self, index):

        SMILES = self.SMILES_list[index]
        sentence = self.sentence_list[index]
        options = self.options_list[index]        
        label = self.labels_list[index]
        df_index = self.index_list[index]

        return SMILES, sentence, label, options, df_index

    def copy(self, idx: Optional[IndexType] = None) -> 'Dataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        data_list = [self.__getitem__(i) for i in idx]

        dataset = copy.copy(self)

        SMILES_list, sentence_list, options_list, labels_list, index_list = list(), list(), list(), list(), list()

        for data in data_list:
            SMILES_list.append(data[0])
            sentence_list.append(data[1])
            labels_list.append(data[2])
            options_list.append(data[3])
            index_list.append(data[4])

        dataset.SMILES_list = SMILES_list
        dataset.sentence_list = sentence_list
        dataset.options_list = options_list
        dataset.labels_list = labels_list
        dataset.index_list = index_list

        return dataset

    def __len__(self):
        return len(self.SMILES_list)


if __name__ == "__main__":

    DATA_PATH = "./data"
    batch_size = 32
    num_workers = 6
    
    dataset = Mol_retrieval_SMILES(DATA_PATH)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")