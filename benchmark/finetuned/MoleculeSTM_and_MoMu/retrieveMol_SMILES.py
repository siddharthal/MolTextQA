import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Mol_retrieval_SMILES
from torch_geometric.loader import DataLoader as pyg_DataLoader

# For Language Models
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertForPreTraining
from utils.bert import prepare_text_tokens


def do_CL_eval(X, Y, label):

    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1) # B, 1, d

    Y = Y.transpose(0, 1)  # B, T, d
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze()  # B*T
    B = X.size()[0]

    pred = logits.argmax(dim=1, keepdim=False)

    return pred, label


class HuggingfaceEncoder:
    def __init__(self, model_name = "laituan245/molt5-large", device = "cuda:0", batch_size=256):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, resume_download=True)
        self.model.to(device)
        self.device = device
        self.batch_size = batch_size
    
    def encode(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(device=self.device, description=text, tokenizer=self.tokenizer, max_seq_len=512)
        out = self.model.encoder(input_ids=text_tokens_ids, attention_mask=text_masks, return_dict=True)["last_hidden_state"][:,0,:]
        return out


class BigModel(nn.Module):
    def __init__(self, main_model, device):
        super(BigModel, self).__init__()
        self.main_model = main_model
        self.device = device
        self.dropout = nn.Dropout(0.1)

        self.main_model.to(self.device)
    
    def encode(self, text):
        text_tokens_ids, text_masks = prepare_text_tokens(
            device=self.device, description=text, tokenizer=tokenizer, max_seq_len=512)

        typ = torch.zeros(text_tokens_ids.shape).long()
        typ = typ.to(self.device)
        pooled_output = self.main_model(text_tokens_ids, token_type_ids=typ, attention_mask=text_masks)['pooler_output']
        logits = self.dropout(pooled_output)

        return logits



@torch.no_grad()
def eval_epoch(dataloader):

    indexs, preds, labels = list(), list(), list()

    for batch in tqdm(dataloader):
    
        text = batch[1]
        label = batch[2]
        molecule_data = batch[3]
        index = batch[4]

        molecule_repr = [model.encode(molecule_data[idx]) for idx in range(5)]
        molecule_repr = torch.stack(molecule_repr)
        text_repr = model.encode(text)

        pred, label = do_CL_eval(text_repr, molecule_repr, label.to(device))
        
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())
        indexs.append(index.detach().cpu())
    
    preds = torch.hstack(preds)
    labels = torch.hstack(labels)
    indexs = torch.hstack(indexs)
    acc = preds.eq(labels).float().mean().item()
    
    return acc, preds, labels, indexs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=3)

    parser.add_argument("--dataspace_path", type=str, default="./data")
    parser.add_argument("--pretrained_model", type=str, default="SciBERT")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=256)

    ########## for BERT model ##########
    parser.add_argument("--max_seq_len", type=int, default=512)

    args = parser.parse_args()
    print("arguments\t", args)

    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = "2"

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    

    ##### prepare text model #####
    if "SciBERT" in args.pretrained_model:
        pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'pretrained_SciBERT')
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
        # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
        model = BigModel(model, device)
        text_dim = 768
    
    elif "KV-PLM" in args.pretrained_model:
        tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        bert_model0 = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
        model = BigModel(bert_model0.bert, device)
        model.load_state_dict(torch.load('./data/pretrained_KV-PLM/ckpt_ret01.pt', map_location = device))

    else:
        model = HuggingfaceEncoder(model_name = args.pretrained_model, device = device, batch_size=256)
    
    dataset = Mol_retrieval_SMILES(args.dataspace_path)

    test_idx = np.where(np.asarray(dataset.split_list) == "test")[0]
    test_set = dataset.copy(test_idx)

    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(test_set, batch_size=args.batch_size, shuffle=False, num_workers = 6)

    test_acc, preds, labels, indexs = eval_epoch(dataloader)
    
    print("arguments\t", args)
    print(test_acc)

    # Write experimental results
    SAVE_PATH = "preds_retMol/"
    os.makedirs(SAVE_PATH, exist_ok=True) # Create directory if it does not exist
    check_dir = SAVE_PATH + args.pretrained_model + ".pth"
    torch.save((indexs, preds, labels), check_dir)

    # Write experimental results
    WRITE_PATH = "results_retMol/"
    os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
    f = open("results_retMol/results.txt", "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write("{}".format(args.pretrained_model))
    f.write("\n")
    f.write("Acc: {}".format(test_acc))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()