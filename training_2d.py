from dlutils.data import DataHandler
from dlutils.training import RegressionTrainer
from plotting import show_parity_plots
import torch
import torch.nn as nn


from scattering_transform.filters import SubNet, FourierSubNetFilters, Morlet
from scattering_transform.scattering_transform import ScatteringTransform2d
from scattering_transform.reducer import Reducer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OSTPV2D(nn.Module):
    def __init__(self, size, num_scales, num_angles):
        super(OSTPV2D, self).__init__()

        self.size = size
        self.num_scales = num_scales

        self.subnet = SubNet(num_ins=4, hidden_sizes=(32, 32), num_outs=1, activation=nn.LeakyReLU)
        self.filters = FourierSubNetFilters(size, num_scales, num_angles, subnet=self.subnet, symmetric=False)
        self.st = ScatteringTransform2d(self.filters)

        self.reducer = Reducer(self.filters, 'none', filters_3d=False)
        self.num_outputs = self.reducer.num_outputs
        self.regressor = nn.Sequential(
            nn.Linear(self.num_outputs, 1)
        )

    def forward(self, x):
        self.filters.update_filters()
        x = self.st(x)
        x = self.reducer(x)
        x = self.regressor(x).squeeze(1)
        return torch.tanh(x)

    def to(self, device):
        super(OSTPV2D, self).to(device)
        self.filters.to(device)
        self.st.to(device)
        self.reducer.to(device)
        self.device = device
        return self

    def init_morlet(self):
        morlet = Morlet(self.size, self.num_scales)
        try:
            morlet.to(self.device)
        except AttributeError:
            pass
        self.filters.initialise_weights(morlet.filter_tensor[:, 0], num_epochs=3000)



def pv_loss(model, data):

    flipped = data.flip((-1,))

    gx = model(data)
    gpx = model(flipped)

    diffs = (gx - gpx).abs()

    mu = diffs.mean()
    sigma = diffs.std()

    return - mu / sigma


def batch_tension_loss(model, data):

    x1 = model(data)
    x2 = model(data.flip((-1,)))

    mean_diff = (x1.mean() - x2.mean()).abs()
    x_sigma_combined = torch.sqrt(x1.std() ** 2 + x2.std() ** 2)
    return -mean_diff / x_sigma_combined


def example_analysis(data_path):

    # load the data
    data = torch.load(data_path)

    # create the data handler
    data_handler = DataHandler(data)
    train_loader, val_loader = data_handler.make_dataloaders(batch_size=64, val_fraction=0.2)

    # create the model
    model = OSTPV2D(size=16, num_scales=3, num_angles=4)
    model.to(device)

    # create the trainer
    trainer = RegressionTrainer(model, train_loader, val_loader, criterion=pv_loss, no_targets=True, device=device)

    # run training and save loss plot
    trainer.run_training(epochs=100, lr=1e-3)
    trainer.loss_plot(save_path='loss_plot.png')

    # show the results
    best_model, best_loss = trainer.get_best_model()
    print("Best loss is {:.3e}".format(best_loss))

    # show the parity plots
    show_parity_plots(best_model, train_loader, val_loader)




