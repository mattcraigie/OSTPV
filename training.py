from data import DataHandler
from dlutils.training import RegressionTrainer
from dlutils.plotting import show_parity_plots
import torch
import torch.nn as nn


from scattering_transform.filters_3d import SubNet3d, FourierSubNetFilters3d, Morlet3d
from scattering_transform.scattering_transform_3d import ScatteringTransform3d
from scattering_transform.reducer import Reducer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OSTPV3D(nn.Module):
    def __init__(self, size, num_scales, init_morlet=True):
        super(OSTPV3D, self).__init__()

        self.subnet = SubNet3d(num_ins=4, hidden_sizes=(32, 32), num_outs=1, activation=nn.LeakyReLU)
        self.filters = FourierSubNetFilters3d(size, num_scales, subnet=self.subnet, symmetric=False)
        self.st = ScatteringTransform3d(self.filters)

        if init_morlet:
            morlet = Morlet3d(size, num_scales)
            self.filters.initialise_weights(morlet.filter_tensor[:, 0], num_epochs=3000)

        self.reducer = Reducer(self.filters, 'none', filters_3d=True)
        self.num_outputs = self.reducer.num_outputs
        self.regressor = nn.Sequential(
            nn.Linear(self.num_outputs, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if self.trainable_filters:
            self.filters.update_filters()
        x = self.st(x)
        x = self.reducer(x)
        x = self.regressor(x).squeeze(1)
        return x

    def to(self, device):
        super(OSTPV3D, self).to(device)
        self.filters.to(device)
        self.st.to(device)
        self.reducer.to(device)
        self.device = device
        return self


def parity_criterion(axes=(-1,)):
    def model_loss(model, data):

        flipped = data.flip(axes)

        gx = model(data)
        gpx = model(flipped)

        mu = (gx - gpx).mean()
        sigma = (gx - gpx).std()

        return mu / sigma

    return model_loss


def analysis():

    # load the data
    data = torch.load('data/mocks_3d.pt')

    # create the data handler
    data_handler = DataHandler(data)
    train_loader, val_loader = data_handler.make_dataloaders(batch_size=128, val_fraction=0.2)

    # create the model
    model = OSTPV3D(size=128, num_scales=4, init_morlet=True)

    # create the trainer
    trainer = RegressionTrainer(model, train_loader, val_loader, criterion=parity_criterion(), no_targets=True, device='cuda')

    # run training and save loss plot
    trainer.run_training(epochs=100, lr=1e-3)
    trainer.loss_plot(save_path='loss_plot.png')

    # show the results
    trainer.load_best()

    # show the parity plots
    show_parity_plots(model, train_loader, val_loader)

