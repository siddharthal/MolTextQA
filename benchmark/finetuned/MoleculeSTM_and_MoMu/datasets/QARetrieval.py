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
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import IndexType
from torch_geometric.loader import DataLoader as pyg_DataLoader

from collections.abc import Sequence

from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import mol_to_graph_data_obj_simple


class QA_retrieval(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        self.qa_csv_file_path = os.path.join(self.root, "QARetrieval", "raw")

        super(QA_retrieval, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    def load_Graph_CID_and_text(self):
        
        self.graphs, self.slices, self.questions, self.options, self.labels_list, self.splits, self.indexs = torch.load(self.processed_paths[0])

        return

    def get_graph(self, index):
        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[index], slices[index + 1])
            data[key] = item[s]
        return data

    def get_for_split(self, index):
        
        data = self.get_graph(index)
        question = self.questions[index]
        options = self.options[index]
        label = self.labels_list[index]
        df_index = self.indexs[index]
        
        return data, question, options, label, df_index

    def get(self, index):
        
        data = self.get_graph(index)
        question = self.questions[index]
        options = self.options[index]
        df_index = self.indexs[index]

        temp_options = list()
        for option in options:
            temp_option = str(question) + " " + str(option)
            temp_options.append(temp_option)

        label = self.labels_list[index]
        
        return data, temp_options, label, df_index

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'QARetrieval', 'processed')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):

        # Read QA Files
        df = pd.read_csv(self.qa_csv_file_path + "/data_splits.csv")

        SMILES_list = df["SMILES"]
        unique_SMILES_list = np.unique(SMILES_list)

        SMILES2Graph = {}
        for SMILES in tqdm(unique_SMILES_list):
            mol = AllChem.MolFromSmiles(SMILES)
            graph = mol_to_graph_data_obj_simple(mol)
            SMILES2Graph[SMILES] = graph
        print("SMILES2graph", len(SMILES2Graph))
        
        SMILES_list, graph_list, question_list, options_list, labels_list, split_list, index_list = list(), list(), list(), list(), list(), list(), list()
        
        for i in tqdm(range(len(df))):

            try:
                SMILES = df["SMILES"][i]
                graph = SMILES2Graph[SMILES]
                question = df["question"][i]
                options = eval(df["options"][i])
                assert len(options) == 5
                label = int(df["correct"][i] - 1)
                assert label < 5
                split = df["split"][i]

                SMILES_list.append(SMILES)
                graph_list.append(graph)
                question_list.append(question)
                options_list.append(options)
                labels_list.append(label)
                split_list.append(split)
                index_list.append(i)
            except:
                pass
        
        print("Total Converted Data: {} / {}".format(len(SMILES_list), len(df)))

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        # Save graphs
        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices, question_list, options_list, labels_list, split_list, index_list), self.processed_paths[0])
        return

    def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':
        r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,
        will clone the full dataset. Otherwise, will only clone a subset of the
        dataset from indices :obj:`idx`.
        Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or
        :obj:`np.ndarray` of type long or bool.
        """
        data_list = [self.get_for_split(i) for i in idx]

        dataset = copy.copy(self)
        dataset._indices = None
        dataset._data_list = None

        graph_list, question_list, options_list, labels_list, index_list = list(), list(), list(), list(), list()

        for data in data_list:
            graph_list.append(data[0])
            question_list.append(data[1])
            options_list.append(data[2])
            labels_list.append(data[3])
            index_list.append(data[4])

        graphs, slices = self.collate(graph_list)
        dataset.graphs, dataset.slices = graphs, slices
        dataset.questions = question_list
        dataset.options = options_list
        dataset.labels_list = labels_list
        dataset.indexs = index_list

        return dataset

    def __len__(self):
        return len(self.labels_list)


if __name__ == "__main__":

    DATA_PATH = "./data"
    batch_size = 32
    num_workers = 6
    
    dataset = QA_retrieval(DATA_PATH)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_idx = np.where(np.asarray(dataset.splits) == "train")[0]
    valid_idx = np.where(np.asarray(dataset.splits) == "valid")[0]
    test_idx = np.where(np.asarray(dataset.splits) == "test")[0]

    train_set = dataset.copy(train_idx)
    valid_set = dataset.copy(valid_idx)
    test_set = dataset.copy(test_idx)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")