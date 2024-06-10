import sys
sys.path.append('.')

import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Call Trainer class
from trainer import trainer

from utils.util import freeze_network, cycle_index
from utils.bert import prepare_text_tokens
from utils import argument


class MoleculeSTM_Trainer(trainer):

    def __init__(self, args):
        trainer.__init__(self, args)

        ##### Build Language Model and Graph Neural Networks #####
        self.build_LM(args)
        self.build_GNN(args)
        
        self.text2latent = nn.Linear(self.text_dim, args.SSL_emb_dim).to(self.device)
        self.mol2latent = nn.Linear(self.molecule_dim, args.SSL_emb_dim).to(self.device)

        ##### Freeze model parameters #####
        if args.representation_frozen:
            freeze_network(self.text_model)
            freeze_network(self.molecule_model)
            model_param_group = [
                {"params": self.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
                {"params": self.mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
            ]
        else:
            model_param_group = [
                {"params": self.text_model.parameters(), "lr": args.text_lr},
                {"params": self.molecule_model.parameters(), "lr": args.mol_lr},
                {"params": self.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
                {"params": self.mol2latent.parameters(), "lr": args.mol_lr * args.mol_lr_scale},
            ]
        
        self.optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
        self.optimal_loss = 1e10


    def train(self):

        # self.evaluate(epoch = 0)

        for epoch in range(1, self.args.epochs + 1):
            
            start_time = time.time()

            accum_loss, accum_acc = 0, 0

            for bc, samples in enumerate(tqdm(self.dataloader)):

                self.optimizer.zero_grad()

                description = samples[0]
                molecule = samples[1]
                
                ##### Forward Pass: Language Model #####
                description_repr = self.get_text_repr(description)

                ##### Forward Pass: Molecule Model #####
                molecule_repr = self.get_molecule_repr(molecule)

                loss_01, acc_01 = self.do_CL(description_repr, molecule_repr)
                loss_02, acc_02 = self.do_CL(molecule_repr, description_repr)

                loss = (loss_01 + loss_02) / 2
                acc = (acc_01 + acc_02) / 2
                
                loss.backward()
                self.optimizer.step()

                accum_loss += loss.item()
                accum_acc += acc
            
            accum_loss /= len(self.dataloader)
            self.writer.add_scalar("loss/Contrastive Loss", accum_loss, epoch)

            accum_acc /= len(self.dataloader)
            self.writer.add_scalar("loss/Contrastive Accuracy", accum_acc, epoch)

            temp_loss = accum_loss
            if temp_loss < self.optimal_loss:
                self.optimal_loss = temp_loss
                self.save_model(epoch=epoch)
            print("[Epoch {}] CL Loss: {:.5f}\tCL Acc: {:.5f}\tTime: {:.5f}".format(epoch, accum_loss, accum_acc, time.time() - start_time))


if __name__ == "__main__":
    
    args, unknown = argument.parse_args()    

    from models import MoleculeSTM_Trainer
    model_trainer = MoleculeSTM_Trainer(args)

    model_trainer.train()