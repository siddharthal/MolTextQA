cd ../
for lr in 1e-3 1e-4 1e-5
do
    python finetune_retMol.py --input_model_config "MoleculeSTM.pth" --device 1 --lr $lr
    python finetune_retMol.py --input_model_config "MoMu.pth" --device 1 --lr $lr
done