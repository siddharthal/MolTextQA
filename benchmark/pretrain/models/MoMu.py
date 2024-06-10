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

from torch_geometric.utils import dropout_adj


class MoMu_Trainer(trainer):

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


    def augmentation(self, molecule, p_aug):

        feat_mask1 = torch.FloatTensor(molecule.x.shape[0]).uniform_() > p_aug
        feat_mask2 = torch.FloatTensor(molecule.x.shape[0]).uniform_() > p_aug

        x1, x2 = molecule.x.clone(), molecule.x.clone()
        x1, x2 = x1 * feat_mask1.reshape(-1, 1), x2 * feat_mask2.reshape(-1, 1)

        molecule1, molecule2 = molecule.clone(), molecule.clone()
        molecule1.x, molecule2.x = x1, x2

        return molecule1, molecule2


    def train(self):

        self.evaluate(epoch = 0)

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
                molecule1, molecule2 = self.augmentation(molecule, 0.5)
                molecule_repr1 = self.get_molecule_repr(molecule1)
                molecule_repr2 = self.get_molecule_repr(molecule2)

                loss_01, acc_01 = self.do_CL(description_repr, molecule_repr1)
                loss_02, acc_02 = self.do_CL(molecule_repr1, description_repr)

                loss_03, acc_03 = self.do_CL(molecule_repr1, molecule_repr2)
                loss_04, acc_04 = self.do_CL(molecule_repr2, molecule_repr1)

                loss = (loss_01 + loss_02 + loss_03 + loss_04) / 4
                acc = (acc_01 + acc_02) / 2
                
                loss.backward()
                if self.args.grad_clip:
                    nn.utils.clip_grad_norm_(list(self.text_model.parameters())+list(self.molecule_model.parameters())
                                             +list(self.text2latent.parameters())+list(self.mol2latent.parameters()),
                                             self.args.max_grad)
                else:
                    pass
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
    args.model = "MoMu"
    args.batch_size = 45

    from models import MoMu_Trainer
    model_trainer = MoMu_Trainer(args)

    model_trainer.train()