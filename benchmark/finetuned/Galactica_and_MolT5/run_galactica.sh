#!/bin/bash
# module load cuda-toolkit/12.2

# training
python llm_qa.py --run=galactica-125m --train
# inference
python llm_qa.py --run=galactica-125m
python llm_qa.py --run=galactica-125m --load_finetuned 

# training
python llm_qa.py --run=galactica-1.3b --train
# inference
python llm_qa.py --run=galactica-1.3b
python llm_qa.py --run=galactica-1.3b --load_finetuned 

# training
python llm_qa.py --run=galactica-6.7b --train
# inference
python llm_qa.py --run=galactica-6.7b
python llm_qa.py --run=galactica-6.7b --load_finetuned 
