import numpy as np
import torch

'''
This code is borrowed from MoleculeSTM for BERT Training
Paper: https://arxiv.org/abs/2212.10789
Code: https://github.com/chao1224/MoleculeSTM
'''

def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)


def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


def prepare_text_tokens(device, description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(tokens_ids).long().to(device)
    masks = torch.Tensor(masks).bool().to(device)
    return tokens_ids, masks