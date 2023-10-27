import torch
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def filtered_hist(x, low, high, bins, smoothing):
    bin_edges = torch.linspace(low, high, bins + 1)
    bin_mids = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    y, _ = torch.histogram(x, bin_edges)
    y_smooth = gaussian_filter1d(y, smoothing)
    return bin_mids, y_smooth


def get_distributions(model, dataloader, bins=100, flip_axes=(-1,), smoothing=3):
    device = model.device
    model.eval()
    with torch.no_grad():
        norms = []
        flips = []
        diffs = []
        for (data,) in dataloader:
            data = data.to(device)
            data_flip = data.flip(dims=flip_axes)
            output = model(data)
            output_flip = model(data_flip)
            norms.append(output.cpu().flatten())
            flips.append(output_flip.cpu().flatten())
            diffs.append((output - output_flip).cpu().flatten())
        norms = torch.cat(norms, dim=0)
        flips = torch.cat(flips, dim=0)
        diffs = torch.cat(diffs, dim=0)

        # pair plotting
        high = max(norms.max().item(), flips.max().item())
        low = min(norms.min().item(), flips.min().item())

        bin_mids_pairs, y_norms = filtered_hist(norms, low, high, bins, smoothing)
        _, y_flips = filtered_hist(flips, low, high, bins, smoothing)

        # difference plotting
        diffs_mean = diffs.mean(0)
        diffs_sigma = diffs.std(0)
        bin_mids_diffs, y_diffs = filtered_hist(diffs, diffs.min(), diffs.max(), bins, smoothing)

        return (bin_mids_pairs, y_norms, y_flips), (bin_mids_diffs, y_diffs, diffs_mean, diffs_sigma)


def pairs_plot(train_left, train_right, train_mids, val_left, val_right, val_mids, save_path=None):
    fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
    axes[0].plot(train_mids, train_left, c='red', label='$g(x)$', linewidth=4)
    axes[0].plot(train_mids, train_right, c='blue', label='$g(Px)$', linewidth=4)
    axes[0].set_title('Train', fontsize=16)
    axes[0].set_xlabel('Value', fontsize=16)
    axes[0].set_ylabel('Number of Patches', fontsize=16)
    # axes[0].set_yticks([])
    # axes[0].set_xticks([])
    axes[0].legend(fontsize=16, loc='upper right')
    axes[0].set_xlim(train_mids.min(), train_mids.max())

    axes[1].plot(val_mids, val_left, c='red', label='Left', linewidth=4)
    axes[1].plot(val_mids, val_right, c='blue', label='Right', linewidth=4)
    axes[1].set_title('Validation', fontsize=16)
    axes[1].set_xlabel('PV Statistic Value', fontsize=16)
    # axes[1].set_yticks([])
    # axes[1].set_xticks([])
    axes[1].set_xlim(val_mids.min(), train_mids.max())

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def diffs_plot(train_diffs, train_mids, val_diffs, val_mids, train_mean, train_sigma, val_mean, val_sigma, save_path=None):

    print("train: {:.3e}, {:.3e} | val: {:.3e}, {:.3e}".format(train_mean, train_sigma, val_mean, val_sigma))

    fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
    axes[0].plot(train_mids, train_diffs, c='red', label='Left Fields', linewidth=4)
    axes[0].set_title('Train', fontsize=16)
    axes[0].set_xlabel('PV Statistic Value', fontsize=16)
    axes[0].set_ylabel('Number of Patches', fontsize=16)
    axes[0].axvline(x=train_mean, c='black', linestyle=':', linewidth=2)
    axes[0].set_yticks([])
    # axes[0].set_xticks([])
    axes[0].legend(fontsize=16, loc='upper right')
    axes[0].set_xlim(train_mids.min(), train_mids.max())

    axes[1].plot(val_mids, val_diffs, c='red', label='Left', linewidth=4)
    axes[1].set_title('Validation', fontsize=16)
    axes[1].set_xlabel('PV Statistic Value', fontsize=16)
    axes[1].axvline(x=val_mean, c='black', linestyle=':', linewidth=2)
    axes[1].set_yticks([])
    # axes[1].set_xticks([])
    axes[1].set_xlim(val_mids.min(), train_mids.max())

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def show_parity_plots(model, train_dataloader, val_dataloader, axes=(-1,)):

    train_pairs, train_diffs = get_distributions(model, train_dataloader, flip_axes=axes)
    val_pairs, val_diffs = get_distributions(model, val_dataloader, flip_axes=axes)

    # Left/right pair parity plot
    train_mids, train_left, train_right = train_pairs
    val_mids, val_left, val_right = val_pairs

    pairs_plot(train_left, train_right, train_mids, val_left, val_right, val_mids)

    # Difference plot
    train_mids, train_diffs, train_mean, train_sigma = train_diffs
    val_mids, val_diffs, val_mean, val_sigma = val_diffs

    diffs_plot(train_diffs, train_mids, val_diffs, val_mids, train_mean, train_sigma, val_mean, val_sigma)

