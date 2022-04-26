import torch
from torch import nn

from NDNT.modules import layers
from NDNT.metrics.mse_loss import MseLoss_datafilter
from copy import deepcopy

class Encoder(nn.Module):
    '''
        Base class for all models.
    '''

    def __init__(self):

        super().__init__()

        self.loss = MseLoss_datafilter()
        self.relu = nn.ReLU()

    def compute_reg_loss(self):
        
        rloss = 0
        if self.stim is not None:
            rloss += self.stim.compute_reg_loss()
        
        if hasattr(self, 'readout_gain'):
            rloss += self.readout_gain.compute_reg_loss()

        if hasattr(self, 'latent_gain'):
            rloss += self.latent_gain.compute_reg_loss()

        if hasattr(self, 'readout_offset'):
            rloss += self.readout_offset.compute_reg_loss()
        
        if hasattr(self, 'latent_offset'):
            rloss += self.latent_offset.compute_reg_loss()
        
        if self.drift is not None:
            rloss += self.drift.compute_reg_loss()
            
        return rloss

    def prepare_regularization(self, normalize_reg = False):
        
        if self.stim is not None:
            self.stim.reg.normalize = normalize_reg
            self.stim.reg.build_reg_modules()

        if hasattr(self, 'readout_gain'):
                self.readout_gain.reg.normalize = normalize_reg
                self.readout_gain.reg.build_reg_modules()
        
        if hasattr(self, 'latent_gain'):
                self.latent_gain.reg.normalize = normalize_reg
                self.latent_gain.reg.build_reg_modules()

        if hasattr(self, 'readout_offset'):
                self.readout_offset.reg.normalize = normalize_reg
                self.readout_offset.reg.build_reg_modules()
        
        if hasattr(self, 'latent_offset'):
                self.latent_offset.reg.normalize = normalize_reg
                self.latent_offset.reg.build_reg_modules()
        
        if self.drift is not None:
            self.drift.reg.normalize = normalize_reg
            self.drift.reg.build_reg_modules()


    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        
        y = batch['robs'][:,self.cids]

        y_hat = self(batch)

        if 'dfs' in batch:
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)

        regularizers = self.compute_reg_loss()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs'][:,self.cids]

        y_hat = self(batch)

        if 'dfs' in batch:
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': None}        

class SharedGain(Encoder):

    def __init__(self, stim_dims,
            NC=None,
            cids=None,
            num_latent=5,
            num_tents=10,
            include_stim=True,
            include_gain=True,
            include_offset=True,
            tents_as_input=False,
            output_nonlinearity='Identity',
            stim_act_func='lin',
            stim_reg_vals={'l2':0.0},
            reg_vals={'l2':0.001},
            act_func='lin'):
        
        super().__init__()

        NCTot = deepcopy(NC)
        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)

        self.stim_dims = stim_dims
        self.name = 'LVM'

        if include_stim:
            self.stim = layers.NDNLayer(input_dims=[stim_dims, 1, 1, 1],
                num_filters=NC,
                NLtype=stim_act_func,
                norm_type=0,
                bias=False,
                reg_vals = stim_reg_vals)
        else:
            self.stim = None


        self.bias = nn.Parameter(torch.zeros(NC, dtype=torch.float32))
        self.output_nl = getattr(nn, output_nonlinearity)()

        ''' neuron drift '''
        if num_tents > 1 and not tents_as_input:
            self.drift = layers.NDNLayer(input_dims=[num_tents, 1, 1, 1],
            num_filters=NC,
            NLtype='lin',
            norm_type=0,
            bias=False,
            reg_vals = reg_vals)
        else:
            self.drift = None

        self.tents_as_input = tents_as_input
        if tents_as_input:
            latent_input_dims = num_tents
        else:
            latent_input_dims = NCTot

        ''' latent variable gain'''
        if include_gain:

            self.latent_gain = layers.NDNLayer(input_dims=[latent_input_dims, 1, 1, 1],
                num_filters=num_latent,
                NLtype='lin',
                norm_type=1,
                bias=False,
                reg_vals = reg_vals)
            self.latent_gain.weight_scale = 1.0

            self.readout_gain = layers.NDNLayer(input_dims=[num_latent, 1, 1, 1],
                num_filters=NC,
                NLtype='lin',
                norm_type=0,
                bias=False,
                reg_vals = reg_vals)
            self.readout_gain.weight_scale = 1.0

        ''' latent variable offset'''
        if include_offset:
            self.latent_offset = layers.NDNLayer(input_dims=[latent_input_dims, 1, 1, 1],
                num_filters=num_latent,
                NLtype='lin',
                norm_type=1,
                bias=False,
                reg_vals = reg_vals)
            self.latent_offset.weight_scale = 1.0

            self.readout_offset = layers.NDNLayer(input_dims=[num_latent, 1, 1, 1],
                num_filters=NC,
                NLtype='lin',
                norm_type=0,
                bias=False,
                reg_vals = reg_vals)
            self.readout_offset.weight_scale = 1.0

    def forward(self, input):
        
        x = 0
        if self.stim is not None:
            x = x + self.stim(input['stim'])
        
        if self.tents_as_input:
            robs = input['tents']
        else:
            robs = input['robs']
            if 'dfs' in input:
                robs = robs * input['dfs']

        if hasattr(self, 'latent_gain'):
            zg = self.latent_gain(robs)
            g = self.readout_gain(zg)
            x = x * self.relu(1 + g)
        
        if hasattr(self, 'latent_offset'):
            zh = self.latent_offset(robs)
            h = self.readout_offset(zh)
            x = x + h

        if self.drift is not None:
            x = x + self.drift(input['tents'])

        x = x + self.bias
        x = self.output_nl(x)
        
        return x