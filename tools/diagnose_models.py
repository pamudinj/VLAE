import torch
import models
import datasets
import numpy as np


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # create dataset
    dataset = datasets.MNIST(batch_size=64, binarize=False, logit_transform=True)
    # get one batch
    data_iter = iter(dataset.train_loader)
    data, _ = next(data_iter)
    data = data.to(device)
    x = dataset.preprocess(data)

    # model configs
    cfg = dict(dataset=dataset, z_dim=50, output_dist='gaussian', x_dim=dataset.dim,
               enc_dim=500, dec_dim=500, svi_lr=5e-4, n_svi_step=4, n_update=1,
               update_lr=0.5, n_flow=4, iaf_dim=500)

    # instantiate MLP VAE
    mlp = models.VAE(**cfg).to(device)
    conv = models.ConvVAE(**cfg).to(device)

    print('MLP params:', count_params(mlp))
    print('Conv params:', count_params(conv))

    # set smaller importance samples for faster test
    models.n_importance_sample = 100

    # compute loss
    mlp_loss = mlp(x)
    conv_loss = conv(x)

    print('MLP loss (train, -ELBO):', mlp_loss.item())
    print('Conv loss (train, -ELBO):', conv_loss.item())

    # importance sample estimate (may be slower)
    with torch.no_grad():
        mlp_is = mlp.importance_sample(x)
        conv_is = conv.importance_sample(x)

    print('MLP importance estimate (sum over batch):', mlp_is.item())
    print('Conv importance estimate (sum over batch):', conv_is.item())


if __name__ == '__main__':
    run()
