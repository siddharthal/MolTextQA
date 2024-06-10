#!/bin/bash
# module load cuda-toolkit/12.2

# training
python llm_qa.py --run=molt5-large --train 
# inference
python llm_qa.py --run=molt5-large 
python llm_qa.py --run=molt5-large --load_finetuned

# training
python llm_qa.py --run=molt5-large-smiles2caption --train
# inference
python llm_qa.py --run=molt5-large-smiles2caption 
python llm_qa.py --run=molt5-large-smiles2caption --load_finetuned

# training
python llm_qa.py --run=molt5-large-caption2smiles --train
# inference
python llm_qa.py --run=molt5-large-caption2smiles 
python llm_qa.py --run=molt5-large-caption2smiles --load_finetuned

