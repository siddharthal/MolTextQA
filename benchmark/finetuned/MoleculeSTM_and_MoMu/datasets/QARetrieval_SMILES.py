import sys
sys.path.append('.')

import copy
import os
from itertools import repeat
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data.dataset import IndexType
from torch_geometric.loader import DataLoader as pyg_DataLoader

from collections.abc import Sequence

from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import mol_to_graph_data_obj_simple


class QA_retrieval_SMILES(Dataset):
    
    def __init__(self, root):
        
        self.root = root
        self.qa_file = os.path.join(self.root, "QARetrieval", "raw", "data_splits.csv")

        df = pd.read_csv(self.qa_file)

        self.SMILES_list, self.question_list, self.options_list, self.labels_list, self.split_list, self.index_list = list(), list(), list(), list(), list(), list()
        
        for i in tqdm(range(len(df))):

            try:
                SMILES = df["SMILES"][i]
                question = df["question"][i]
                options = eval(df["options"][i])
                assert len(options) == 5
                label = int(df["correct"][i] - 1)
                split = df["split"][i]

                self.SMILES_list.append(SMILES)
                self.question_list.append(question)
                self.options_list.append(options)
                self.labels_list.append(label)
                self.split_list.append(split)
                self.index_list.append(i)
            except:
                pass
    

    def __split__(self, index):

        SMILES = self.SMILES_list[index]
        question = self.question_list[index]
        options = self.options_list[index]
        label = self.labels_list[index]
        df_index = self.index_list[index]

        return SMILES, question, options, label, df_index


    def __getitem__(self, index):

        SMILES = self.SMILES_list[index]
        question = self.question_list[index]
        options = self.options_list[index]

        temp_options = list()
        for option in options:
            temp_option = str(question) + " " + str(option)
            temp_options.append(temp_option)

        label = self.labels_list[index]
        df_index = self.index_list[index]

        return SMILES, temp_options, label, df_index
    

    def copy(self, idx: Optional[IndexType] = None) -> 'Dataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        data_list = [self.__split__(i) for i in idx]

        dataset = copy.copy(self)

        SMILES_list, question_list, options_list, labels_list, index_list = list(), list(), list(), list(), list()

        for data in data_list:
            SMILES_list.append(data[0])
            question_list.append(data[1])
            options_list.append(data[2])
            labels_list.append(data[3])
            index_list.append(data[4])

        dataset.SMILES_list = SMILES_list
        dataset.question_list = question_list
        dataset.options_list = options_list
        dataset.labels_list = labels_list
        dataset.index_list = index_list

        return dataset

    def __len__(self):
        return len(self.labels_list)


if __name__ == "__main__":

    DATA_PATH = "./data"
    batch_size = 32
    num_workers = 6
    
    dataset = QA_retrieval_SMILES(DATA_PATH)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_idx = np.where(np.asarray(dataset.split_list) == "test")[0]
    test_set = dataset.copy(test_idx)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")