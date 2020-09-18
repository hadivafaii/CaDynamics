import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_utils import print_num_params
from generic_utils import to_np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


class AutoEncoder(nn.Module):
    def __init__(self,
                 nb_units,
                 nb_cells,
                 lr: float = 1e-3,
                 wd: float = 1e-2,
                 tmax: int = 10,
                 eta_min: float = 1e-6,
                 verbose=False, ):
        super(AutoEncoder, self).__init__()

        encoder_units = [nb_cells] + nb_units
        encoder_layers = []
        for i in range(len(nb_units)):
            encoder_layers.extend([nn.Linear(encoder_units[i], encoder_units[i + 1]), nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*encoder_layers[:-1])

        decoder_units = encoder_units[::-1]
        decoder_layers = []
        for i in range(len(nb_units)):
            decoder_layers.extend([nn.Linear(decoder_units[i], decoder_units[i + 1]), nn.ReLU(inplace=True)])
        self.decoder = nn.Sequential(*decoder_layers[:-1])

        self.criterion = nn.MSELoss(reduction="sum")

        self.optim = None
        self.optim_schedule = None
        self._setup_optim(lr, wd, tmax, eta_min)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y

    def trn(self, data, batch_size=32, epoch=0):
        self.train()

        max_num_batches = int(np.ceil(len(data) / batch_size))
        pbar = tqdm(range(max_num_batches))
        cuml_loss = 0.0

        for b in pbar:
            start = b * batch_size
            end = min((b + 1) * batch_size, len(data))

            x = data[range(start, end)]
            x = torch.from_numpy(x).float().cuda()
            _, y = self(x)

            loss = self.criterion(y, x) / (end - start)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            msg = "epoch: {}, loss: {:.3f}"
            msg = msg.format(epoch, loss.item())
            pbar.set_description(msg)
            cuml_loss += loss.item()
            if (b + 1) == max_num_batches:
                msg = "epoch # {}. avg loss: {:.4f}"
                msg = msg.format(epoch, cuml_loss / max_num_batches)
                pbar.set_description(msg)

        self.optim_schedule.step()

    def tst(self, data, epoch=-1):
        self.eval()

        x = torch.from_numpy(data).float().cuda()
        with torch.no_grad():
            z, y = self(x)

        loss = self.criterion(y, x) / len(data)

        x_np = to_np(x)
        y_np = to_np(y)
        z_np = to_np(z)

        r2 = r2_score(x_np, y_np, multioutput='raw_values') * 100
        r2_plus = np.maximum(r2, 0)

        plt.plot(r2_plus)
        msg = "epoch # {},   mean r2 = {:.2f} {:s},   tst loss = {:.4f}"
        msg = msg.format(epoch, r2_plus.mean(), "%", loss.item())
        plt.title(msg)
        plt.show()

        return r2, x_np, y_np, z_np

    def _setup_optim(self, lr, wd, tmax, eta_min):
        self.optim = AdamW(self.parameters(), lr=lr, weight_decay=wd)
        self.optim_schedule = CosineAnnealingLR(self.optim, T_max=tmax, eta_min=eta_min)
