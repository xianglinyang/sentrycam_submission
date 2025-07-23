from abc import ABC, abstractmethod
import os
import time
import gc 
import json
from tqdm import tqdm
import itertools
import torch

"""
1. construct a spatio-temporal complex
2. construct an edge-dataset
3. train the network

Trainer should contains
1. train_step function
2. early stop
3. ...
"""

class TrainerAbstractClass(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def reset_optim(self):
        pass

    @abstractmethod
    def update_edge_loader(self):
        pass

    @abstractmethod
    def update_vis_model(self):
        pass

    @abstractmethod
    def update_optimizer(self):
        pass

    @abstractmethod
    def update_lr_scheduler(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def train(self):
       pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def record_time(self):
        pass

    @abstractmethod
    def log(self):
        pass

    @abstractmethod
    def read(self):
        pass




class SingleVisTrainer(TrainerAbstractClass):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.DEVICE = DEVICE
        self.edge_loader = edge_loader
        self._loss = {
            'loss': list(),
            'umap': list(),
            'recon':list()
        }

    @property
    def loss(self):
        return self._loss

    def reset_optim(self, optim, lr_s):
        self.optimizer = optim
        self.lr_scheduler = lr_s
        print("Successfully reset optimizer!")
    
    def update_edge_loader(self, edge_loader):
        del self.edge_loader
        gc.collect()
        self.edge_loader = edge_loader
    
    def update_vis_model(self, model):
        self.model.load_state_dict(model.state_dict())
    
    def update_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def update_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def train_step(self, verbose=1):
        self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))

        # for data in self.edge_loader:
        for data in t:
            edge_to, edge_from, a_to, a_from = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            
            embedding_to, recon_to = self.model(edge_to)
            embedding_from, recon_from = self.model(edge_from)
            
            outputs = dict()
            outputs["umap"] = (embedding_to, embedding_from)
            outputs["recon"] = (recon_to, recon_from)
            # outputs = self.model(edge_to, edge_from)
            
            umap_l, recon_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # record loss history
        self._loss['loss'].append(sum(all_loss) / len(all_loss))
        self._loss['umap'].append(sum(umap_losses) / len(umap_losses))
        self._loss['recon'].append(sum(recon_losses) / len(recon_losses))
        self.model.eval()
        if verbose:
            message = (f"umap:{self._loss['umap'][-1]:.4f}\trecon:{self._loss['recon'][-1]:.4f}\tloss:{self._loss['loss'][-1]:.4f}"
                    )
            print(message)
        return self.loss

    def train(self, PATIENT, MAX_EPOCH_NUMS):
        patient = PATIENT
        time_start = time.time()
        for epoch in range(MAX_EPOCH_NUMS):
            print("====================\nepoch:{}\n===================".format(epoch+1))
            prev_loss = self.loss['loss'][-1] if len(self.loss['loss'])>0 else 100
            loss = self.train_step()['loss'][-1]
            self.lr_scheduler.step()
            # early stop, check whether converge or not
            if prev_loss - loss < 1E-2:
                if patient == 0:
                    break
                else:
                    patient -= 1
            else:
                patient = PATIENT

        time_end = time.time()
        time_spend = time_end - time_start
        print("Time spend: {:.2f} for training vis model...".format(time_spend))
        return epoch+1, round(time_spend, 3)

    def load(self, file_path):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = torch.load(file_path, map_location="cpu")
        self._loss = save_model["loss"]
        self.model.load_state_dict(save_model["state_dict"])
        self.model.to(self.DEVICE)
        print("Successfully load visualization model...")

    def save(self, save_dir, file_name):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = {
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()}
        save_path = os.path.join(save_dir, file_name + '.pth')
        torch.save(save_model, save_path)
        print("Successfully save visualization model...")
    
    def record_time(self, save_dir, file_name, key, epoch, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if epoch is None:
            evaluation[key] = t
        else:
            if key not in evaluation.keys():
                evaluation[key] = dict()
            evaluation[key][str(epoch)] = t
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)
    
    def log(self, log_dir, epoch):
        log_path = os.path.join(log_dir, "log.json")
        curr_log = self.read(log_dir)
        curr_log.append(epoch)
        with open(log_path, "w") as f:
            json.dump(curr_log, f)
    
    def read(self, log_dir):
        log_path = os.path.join(log_dir, "log.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                curr_log = json.load(f)
        else:
            curr_log = list()
        curr_log.sort()
        return curr_log


class HybridVisTrainer(SingleVisTrainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE):
        super().__init__(model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE)
        self._loss['smooth'] = list()

    def train_step(self, verbose=1):
        self.model = self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        smooth_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))
        
        for data in t:
            edge_to, edge_from, a_to, a_from, embedded_to, coeffi_to = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            embedded_to = embedded_to.to(device=self.DEVICE, dtype=torch.float32)
            coeffi_to = coeffi_to.to(device=self.DEVICE, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            umap_l, recon_l, smooth_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, embedded_to, coeffi_to, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            smooth_losses.append(smooth_l.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._loss['loss'].append(sum(all_loss) / len(all_loss))
        self._loss['umap'].append(sum(umap_losses) / len(umap_losses))
        self._loss['recon'].append(sum(recon_losses) / len(recon_losses))
        self._loss['smooth'].append(sum(smooth_losses) / len(smooth_losses))
        self.model.eval()
        if verbose:
            message = (f"umap:{self._loss['umap'][-1]:.4f}\trecon:{self._loss['recon'][-1]:.4f}\tsmooth:{self._loss['smooth'][-1]:.4f}\tloss:{self._loss['loss'][-1]:.4f}\t"
                       )
            print(message)
        return self.loss
    
    def record_time(self, save_dir, file_name, operation, seg, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if operation not in evaluation.keys():
            evaluation[operation] = dict()
        evaluation[operation][str(seg)] = round(t, 3)
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)
        

class DVITrainer(SingleVisTrainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE):
        super().__init__(model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE)
        self._loss["temporal"] = list()
    
    def train_step(self, verbose=1):
        self.model = self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        temporal_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))
        
        for data in t:
            edge_to, edge_from, a_to, a_from = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)

            # outputs = self.model(edge_to, edge_from)
            embedding_to, recon_to = self.model(edge_to)
            embedding_from, recon_from = self.model(edge_from)
            
            outputs = dict()
            outputs["umap"] = (embedding_to, embedding_from)
            outputs["recon"] = (recon_to, recon_from)
            
            umap_l, recon_l, temporal_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, self.model, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            temporal_losses.append(temporal_l.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._loss['loss'].append(sum(all_loss) / len(all_loss))
        self._loss['umap'].append(sum(umap_losses) / len(umap_losses))
        self._loss['recon'].append(sum(recon_losses) / len(recon_losses))
        self._loss['temporal'].append(sum(temporal_losses) / len(temporal_losses))
        self.model.eval()
        if verbose:
            message = f"umap:{self._loss['umap'][-1]:.4f}\trecon:{self._loss['recon'][-1]:.4f}\ttemporal:{self._loss['temporal'][-1]:.4f}\tloss:{self._loss['loss'][-1]:.4f}\t"
            print(message)
        return self.loss
    
    def record_time(self, save_dir, file_name, operation, iteration, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if operation not in evaluation.keys():
            evaluation[operation] = dict()
        evaluation[operation][iteration] = round(t, 3)
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)

class LocalTemporalTrainer(SingleVisTrainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE):
        super().__init__(model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE)
        self._loss['smooth']=list()
    
    def train_step(self, verbose=1):
        self.model = self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        smooth_losses = []
        umap_losses = []
        recon_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader))
        
        for data in t:
            edge_to, edge_from, a_to, a_from, coeffi_from, embedded_from = data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            coeffi_from = coeffi_from.to(device=self.DEVICE, dtype=torch.bool)
            embedded_from = embedded_from.to(device=self.DEVICE, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            # embedding_to, recon_to = self.model(edge_to)
            # embedding_from, recon_from = self.model(edge_from)
            
            # outputs = dict()
            # outputs["umap"] = (embedding_to, embedding_from)
            # outputs["recon"] = (recon_to, recon_from)

            umap_l, recon_l, smooth_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, coeffi_from, embedded_from, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            smooth_losses.append(smooth_l.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self._loss['loss'].append(sum(all_loss) / len(all_loss))
        self._loss['umap'].append(sum(umap_losses) / len(umap_losses))
        self._loss['recon'].append(sum(recon_losses) / len(recon_losses))
        self._loss['smooth'].append(sum(smooth_losses) / len(smooth_losses))
        self.model.eval()
        if verbose:
            message = f"umap:{self._loss['umap'][-1]:.4f}\trecon:{self._loss['recon'][-1]:.4f}\tsmooth:{self._loss['smooth'][-1]:.4f}\tloss:{self._loss['loss'][-1]:.4f}\t"
            print(message)
        return self.loss
    
    def record_time(self, save_dir, file_name, operation, iteration, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if operation not in evaluation.keys():
            evaluation[operation] = dict()
        evaluation[operation][iteration] = t
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)


class SplitTemporalTrainer(SingleVisTrainer):
    def __init__(self, model, criterion, optimizer, lr_scheduler, spatial_edge_loader, temporal_edge_loader, DEVICE):
        super().__init__(model, criterion, optimizer, lr_scheduler, spatial_edge_loader, DEVICE)
        self.temporal_edge_loader = temporal_edge_loader
        self._loss['temporal']=list()
        # self._loss['temporal_recon']=list()

    def update_edge_loader(self, spatial_edge_loader, temporal_edge_loader):
        del self.spatial_edge_loader
        del self.temporal_edge_loader
        gc.collect()
        self.spatial_edge_loader = spatial_edge_loader
        self.temporal_edge_loader = temporal_edge_loader
        
    def train_step(self, verbose=1):
        self.model.to(device=self.DEVICE)
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        temporal_losses = []
        # temporal_recon_losses = []
        # iterate the shorter one until we iterate through longest one 
        if len(self.edge_loader)>len(self.temporal_edge_loader):
            t = tqdm(zip(self.edge_loader, itertools.cycle(self.temporal_edge_loader)), total=len(self.edge_loader), leave=True)
        else:
            t = tqdm(zip(itertools.cycle(self.edge_loader), self.temporal_edge_loader), total=len(self.temporal_edge_loader), leave=True)
        for spatial_data, temporal_data in t:
            edge_to, edge_from, a_to, a_from = spatial_data
            edge_t_to, edge_t_from, embedded_from, margins = temporal_data

            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)
            a_to = a_to.to(device=self.DEVICE, dtype=torch.float32)
            a_from = a_from.to(device=self.DEVICE, dtype=torch.float32)
            
            edge_t_to = edge_t_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_t_from = edge_t_from.to(device=self.DEVICE, dtype=torch.float32)
            embedded_from = embedded_from.to(device=self.DEVICE, dtype=torch.float32)
            margins = margins.to(device=self.DEVICE, dtype=torch.float32)
            
            embedding_to, recon_to = self.model(edge_to)
            embedding_from, recon_from = self.model(edge_from)
            embedding_t_to, _ = self.model(edge_t_to)
            embedding_t_from, _ = self.model(edge_t_from)
            
            outputs = dict()
            outputs["umap"] = (embedding_to, embedding_from)
            outputs["recon"] = (recon_to, recon_from)
            outputs['temporal'] = (embedding_t_to, embedding_t_from)
            # outputs['temporal_recon'] = (edge_t_to, edge_t_from, recon_t_to, recon_t_from)
            
            umap_l, recon_l, temporal_l, loss = self.criterion(edge_to, edge_from, a_to, a_from, embedded_from, margins, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            temporal_losses.append(temporal_l.item())
            # temporal_recon_losses.append(t_recon_l.item())
            # ===================backward====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # record loss history
        self._loss['loss'].append(sum(all_loss) / len(all_loss))
        self._loss['umap'].append(sum(umap_losses) / len(umap_losses))
        self._loss['recon'].append(sum(recon_losses) / len(recon_losses))
        self._loss['temporal'].append(sum(temporal_losses) / len(temporal_losses))
        # self._loss['temporal_recon'].append(sum(temporal_recon_losses) / len(temporal_recon_losses))
        self.model.eval()
        if verbose:
            message = (f"umap:{self._loss['umap'][-1]:.4f}\trecon:{self._loss['recon'][-1]:.4f}\ttemporal:{self._loss['temporal'][-1]:.4f}\tloss:{self._loss['loss'][-1]:.4f}")
            print(message)
        return self.loss
    
    def record_time(self, save_dir, file_name, operation, iteration, t):
        # save result
        save_file = os.path.join(save_dir, file_name+".json")
        if not os.path.exists(save_file):
            evaluation = dict()
        else:
            f = open(save_file, "r")
            evaluation = json.load(f)
            f.close()
        if operation not in evaluation.keys():
            evaluation[operation] = dict()
        evaluation[operation][iteration] = t
        with open(save_file, 'w') as f:
            json.dump(evaluation, f)