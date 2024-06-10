import os
import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.util import cycle_index, save

from tensorboardX import SummaryWriter

# Call dataset
from datasets import PubChemSTM_Datasets_Graph, PubChemSTM_SubDatasets_Graph
from torch_geometric.loader import DataLoader as pyg_DataLoader

from utils.argument import config2string
from utils.bert import prepare_text_tokens

# For Language Models
from transformers import AutoModel, AutoTokenizer

# For Graph Neural Networks
from layers import GNN, GNN_graphpred

class trainer:
    
    def __init__(self, args):
        
        self.args = args

        d = datetime.datetime.now()
        date = d.strftime("%x")[-2:] + d.strftime("%x")[0:2] + d.strftime("%x")[3:5]

        self.config_str = "{}_".format(date) + config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
        # Use Tensorboard
        if args.writer:
            self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))
        else:
            self.writer = SummaryWriter(log_dir="runs_/{}".format(self.config_str))
        
        os.environ['TOKENIZERS_PARALLELISM'] = 'False'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(args.device)
        # torch.cuda.set_device(self.device)
        # self.device = "cpu"
        
        ##### Load Train Data #####
        dataloader_class = pyg_DataLoader
        if args.dataset == "PubChemSTM":
            self.dataset = PubChemSTM_Datasets_Graph(args.data_path)
        elif args.dataset == "PubChemSTM1K":
            self.dataset = PubChemSTM_SubDatasets_Graph(args.data_path, size=1000)
        elif args.dataset == "PubChemSTM10K":
            self.dataset = PubChemSTM_SubDatasets_Graph(args.data_path, size=10000)
        elif args.dataset == "PubChemSTM100K":
            self.dataset = PubChemSTM_SubDatasets_Graph(args.data_path, size=100000)

        self.dataloader = dataloader_class(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    
    def build_LM(self, args):
        """
        Build Language Models for Encoding Textual Description
        """
        if args.lm == "SciBERT":
            pretrained_SciBERT_folder = os.path.join(args.data_path, 'pretrained_SciBERT')
            self.text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
            self.text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(self.device)
            self.text_dim = 768
        else:
            raise Exception


    def build_GNN(self, args):
        """
        Build Graph Neural Networks for Encoding Molecular Graph Structure
        """
        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)

        self.molecule_model = GNN_graphpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model)

        pretrained_model_path = os.path.join(args.data_path, "pretrained_GraphMVP", args.pretrain_gnn_mode, "model.pth")
        self.molecule_model.from_pretrained(pretrained_model_path, self.device)
        self.molecule_model.to(self.device)
        self.molecule_dim = args.gnn_emb_dim
    

    def get_text_repr(self, text):
        """
        Get representation of molecular textual description with Language Models
        """
        text_tokens_ids, text_masks = prepare_text_tokens(
            device=self.device, description=text, tokenizer=self.text_tokenizer, max_seq_len=self.args.max_seq_len)
        text_output = self.text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
        text_repr = text_output["pooler_output"]
        text_repr = self.text2latent(text_repr)
        
        return text_repr
    

    def get_molecule_repr(self, molecule):
        """
        Get representation of molecules with Graph Neural Networks
        """        
        molecule_output, _ = self.molecule_model(molecule.to(self.device))
        molecule_repr = self.mol2latent(molecule_output)

        return molecule_repr


    def do_CL(self, X, Y):
        """
        Calculate the Contrastive Loss for Model Training
        """

        if self.args.normalize:
            X = F.normalize(X, dim=-1)
            Y = F.normalize(Y, dim=-1)

        if self.args.SSL_loss == 'EBM_NCE':
            criterion = nn.BCEWithLogitsLoss()
            neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(self.args.CL_neg_samples)], dim=0)
            neg_X = X.repeat((self.args.CL_neg_samples, 1))

            pred_pos = torch.sum(X * Y, dim=1) / self.args.T
            pred_neg = torch.sum(neg_X * neg_Y, dim=1) / self.args.T

            loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
            loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
            CL_loss = (loss_pos + self.args.CL_neg_samples * loss_neg) / (1 + self.args.CL_neg_samples)

            CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                    (len(pred_pos) + len(pred_neg))
            CL_acc = CL_acc.detach().cpu().item()
        
        else :
            criterion = nn.CrossEntropyLoss()
            B = X.size()[0]
            logits = torch.mm(X, Y.transpose(1, 0))  # B*B
            logits = torch.div(logits, self.args.T)
            labels = torch.arange(B).long().to(logits.device)  # B*1

            CL_loss = criterion(logits, labels)
            pred = logits.argmax(dim=1, keepdim=False)
            CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

        return CL_loss, CL_acc


    def save_model(self, epoch = None):

        save(self.args.checkpoint_path, "text", self.text_model, self.config_str)
        save(self.args.checkpoint_path, "molecule", self.molecule_model, self.config_str)

        if self.text2latent is not None:
            save(self.args.checkpoint_path, "text2latent", self.text2latent, self.config_str)
        
        if self.mol2latent is not None:
            save(self.args.checkpoint_path, "mol2latent", self.mol2latent, self.config_str)
