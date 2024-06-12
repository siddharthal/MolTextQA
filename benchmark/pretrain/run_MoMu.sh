cd ../
python pretrain.py --lm SciBERT --epochs 30 --writer True --text_lr 1e-5 --mol_lr 1e-5 --model MoMu --dataset PubChemSTM --data_path ./data/PubChemSTM --num_workers 2 --T 0.1 --batch_size 45 --device 0