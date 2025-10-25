import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import distribution
import torch.autograd.functional as autograd_f


class BaseEncoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.linear_hidden0 = nn.Linear(self.x_dim[1] * self.x_dim[2] * self.x_dim[0], h_dim)
        self.linear_hidden1 = nn.Linear(h_dim, h_dim)
        self.linear_mu = nn.Linear(h_dim, z_dim)
        self.linear_logvar = nn.Linear(h_dim, z_dim)
        self.activation = F.relu

    def forward(self, x):
        h = self.activation(self.linear_hidden0(x))
        h = self.activation(self.linear_hidden1(h))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)

        return distribution.DiagonalGaussian(mu, logvar), h

class BaseDecoder(nn.Module):
    def __init__(self, z_dim, output_dist, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.linear_hidden0 = nn.Linear(z_dim, h_dim)
        self.linear_hidden1 = nn.Linear(h_dim, h_dim)
        self.linear_mu = nn.Linear(h_dim, self.x_dim[1] * self.x_dim[2] * self.x_dim[0])
        self.activation = F.relu
        self.output_dist = output_dist

        if output_dist == 'gaussian':
            self.logvar = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, z, compute_jacobian=False):
        if compute_jacobian:
            h = self.activation(self.linear_hidden0(z))

            # activation_mask: [batch_size, hidden_dim, 1]
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float)
            # W: [hidden_dim, input_dim]
            W = self.linear_hidden0.weight
            # W: [batch_size, hidden_dim, input_dim]
            W = activation_mask * W

            h = self.activation(self.linear_hidden1(h))
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float)
            W = torch.matmul(self.linear_hidden1.weight, W)
            W = activation_mask * W

            W = torch.matmul(self.linear_mu.weight, W)

            mu = self.linear_mu(h)
            W_out = W

            if self.output_dist == 'gaussian':
                return distribution.DiagonalGaussian(mu, self.logvar), W_out
            elif self.output_dist == 'bernoulli':
                mu = torch.sigmoid(mu)
                mu_clip = torch.clamp(mu, min=1e-5, max=1.0 - 1e-5)
                self.logvar = -torch.log((mu_clip * (1 - mu_clip)) + 1e-5)
                return distribution.Bernoulli(mu), W_out
            else:
                raise ValueError

        else:
            h = self.activation(self.linear_hidden0(z))
            h = self.activation(self.linear_hidden1(h))
            mu = self.linear_mu(h)

            if self.output_dist == 'gaussian':
                return distribution.DiagonalGaussian(mu, self.logvar)
            elif self.output_dist == 'bernoulli':
                mu = torch.sigmoid(mu)
                return distribution.Bernoulli(mu)
            else:
                raise ValueError


class HouseHolderFlow(nn.Module):
    def __init__(self, h_dim, z_dim, n_flow):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.linear_flows = nn.ModuleList([nn.Linear(h_dim, z_dim) if i == 0
                                           else nn.Linear(z_dim, z_dim)
                                           for i in range(n_flow)])
        self.I = torch.eye(self.z_dim).unsqueeze(0).cuda()

    def forward(self, h):
        H = self.I.clone()
        for linear_flow in self.linear_flows:
            h = linear_flow(h)
            H = torch.matmul(self.I - 2.0 * torch.matmul(h.unsqueeze(-1),
                                                         h.unsqueeze(1))
                             / h.pow(2).sum(dim=-1, keepdim=True).unsqueeze(-1),
                             H)

        return H


class MADE(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        2-Layer MADE (http://arxiv.org/abs/1502.03509)
        Used as AutoregressiveNN for Inverse Autogressive Flow (IAF)
        """

        super(MADE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(self.input_size + self.hidden_size, self.hidden_size, bias=False)
        self.b = nn.Parameter(torch.randn(self.hidden_size))

        self.V_s = nn.Linear(self.hidden_size, self.input_size, bias=False)
        self.V_m = nn.Linear(self.hidden_size, self.input_size, bias=False)
        self.c_s = nn.Parameter(torch.ones(self.input_size) * 2.0)
        self.c_m = nn.Parameter(torch.randn(self.input_size))

        self.W_mask, self.V_mask = self.generate_mask()

        self.relu = nn.PReLU()

    def generate_mask(self):
        """Generate masks for network weights"""
        # m(k)
        # randomly generate the indexes.
        # Q: shouldn't this be input_size - 1? A: it is. arg high discounts itself.
        max_masks = np.random.randint(low=1, high=self.input_size, size=self.hidden_size)

        # M^W
        # note: input_size + hidden_size b/c z and h are concatted as the input.
        W_mask = np.fromfunction(
            lambda k, d: max_masks[k] >= d + 1, (self.hidden_size, self.input_size + self.hidden_size),
            dtype=int).astype(np.float32)
        W_mask = nn.Parameter(torch.from_numpy(W_mask), requires_grad=False)

        # M^V
        V_mask = np.fromfunction(
            lambda d, k: d + 1 > max_masks[k], (self.input_size, self.hidden_size),
            dtype=int).astype(np.float32)
        V_mask = nn.Parameter(torch.from_numpy(V_mask), requires_grad=False)

        # Check strict lower triangular
        # M^V @ M^W must be strictly lower triangular
        # => M^V @ M^W 's upper triangular is zero matrix
        assert ((V_mask.data @ W_mask.data).triu().eq(
            torch.zeros(self.input_size, self.input_size + self.hidden_size))).all()

        return W_mask, V_mask

    def apply_mask(self):
        """Mask weights"""
        self.W.weight.data = (self.W.weight * self.W_mask).data
        self.V_s.weight.data = (self.V_s.weight * self.V_mask).data
        self.V_m.weight.data = (self.V_m.weight * self.V_mask).data

    def forward(self, z, h):
        """
        Args:
            z: [batch_size, z_size]
            h: [batch_size, h_size]
            input_size = z_size + h_size
        Return
            m: [batch_size, input_size]
            s: [batch_size, input_size]
        """

        self.apply_mask()
        x = self.W(torch.cat([z, h], dim=1)) + self.b
        x = self.relu(x)

        m = self.V_m(x) + self.c_m
        s = self.V_s(x) + self.c_s

        return m, s


class ConvEncoder(nn.Module):
    """Simple convolutional encoder that mirrors the MLP BaseEncoder.
    Returns a DiagonalGaussian distribution and a hidden vector h.
    """
    def __init__(self, z_dim, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim

        c = x_dim[0]
        # conv layers: 28x28 -> 14x14 -> 7x7
        self.conv1 = nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        conv_out_dim = 64 * (x_dim[1] // 4) * (x_dim[2] // 4)
        self.fc1 = nn.Linear(conv_out_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.linear_mu = nn.Linear(h_dim, z_dim)
        self.linear_logvar = nn.Linear(h_dim, z_dim)
        self.activation = F.relu

    def forward(self, x):
        # Accept either flattened input [batch, image_size] (as produced by
        # dataset.preprocess) or image tensor [batch, C, H, W]. If flattened,
        # reshape back to [B,C,H,W] using self.x_dim.
        if x.dim() == 2:
            batch = x.size(0)
            x = x.view(batch, self.x_dim[0], self.x_dim[1], self.x_dim[2])
        # x: [batch, C, H, W]
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        h_flat = h.view(h.size(0), -1)
        h2 = self.activation(self.fc1(h_flat))
        h3 = self.activation(self.fc2(h2))
        mu = self.linear_mu(h3)
        logvar = self.linear_logvar(h3)
        return distribution.DiagonalGaussian(mu, logvar), h3


class ConvDecoder(nn.Module):
    """Convolutional decoder that mirrors BaseDecoder behaviour.
    If compute_jacobian=True it computes the Jacobian of output mu wrt z
    using autograd.functional.jacobian. This is slow and intended for
    evaluation / VLAE use with small batches.
    """
    def __init__(self, z_dim, output_dist, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.output_dist = output_dist
        self.x_dim = x_dim
        self.h_dim = h_dim

        # project z to a feature map of size 64 x (H/4) x (W/4)
        conv_h = x_dim[1] // 4
        conv_w = x_dim[2] // 4
        conv_feat = 64 * conv_h * conv_w

        self.fc0 = nn.Linear(z_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, conv_feat)
        # transpose convs to upsample 7x7 -> 14x14 -> 28x28
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, x_dim[0], kernel_size=4, stride=2, padding=1)
        self.activation = F.relu

        if output_dist == 'gaussian':
            self.logvar = nn.Parameter(torch.Tensor([0.0]))

    def _decode_mu(self, z):
        h = self.activation(self.fc0(z))
        h = self.activation(self.fc1(h))
        # reshape to [batch, 64, H/4, W/4]
        batch = z.size(0)
        conv_h = self.x_dim[1] // 4
        conv_w = self.x_dim[2] // 4
        h_map = h.view(batch, 64, conv_h, conv_w)
        h = self.activation(self.deconv1(h_map))
        mu = self.deconv2(h)
        mu_flat = mu.view(batch, -1)
        return mu_flat

    def forward(self, z, compute_jacobian=False):
        mu_flat = self._decode_mu(z)

        if compute_jacobian:
            # autograd.functional.jacobian per-sample (slow)
            batch = z.size(0)
            # autograd.functional.jacobian per-sample (slow)
            # If batch is large, compute Jacobians in smaller chunks to avoid extremely long stalls
            CHUNK = 16
            if batch > CHUNK:
                print(f"WARNING: compute_jacobian=True with batch={batch} is slow; computing Jacobian in chunks of {CHUNK} to avoid long stall.")

            z_req = z.detach().requires_grad_(True)

            # Prefer functorch/vmap + jacrev for batched jacobian if available
            use_functorch = False
            try:
                # functorch may be available as a separate package or under torch.func
                try:
                    from functorch import jacrev, vmap  # type: ignore
                except Exception:
                    # PyTorch >=2.0 has torch.func.jacrev and torch.vmap
                    from torch.func import jacrev  # type: ignore
                    from torch import vmap  # type: ignore
                use_functorch = True
            except Exception:
                use_functorch = False

            W_out_chunks = []
            for start in range(0, batch, CHUNK):
                end = min(batch, start + CHUNK)
                zb = z_req[start:end]

                if use_functorch:
                    # compute jacobian for the chunk in a batched, vectorized way
                    try:
                        # jacrev expects a function that maps a single sample to outputs
                        # define single-sample function
                        def f_single(z_single):
                            return self._decode_mu(z_single.unsqueeze(0)).squeeze(0)

                        # vmap over jacrev to get [chunk, image_size, z_dim]
                        from functorch import vmap, jacrev  # type: ignore
                        J_chunk = vmap(jacrev(f_single))(zb)
                        W_out_chunks.append(J_chunk)
                        continue
                    except Exception:
                        # Fall back to autograd per-sample within this chunk
                        pass

                # compute jacobian for each sample in this chunk using autograd.functional
                for b in range(zb.size(0)):
                    def f(zb_single):
                        return self._decode_mu(zb_single.unsqueeze(0)).squeeze(0)

                    J = autograd_f.jacobian(f, zb[b])
                    W_out_chunks.append(J)

            # concatenate chunks
            if len(W_out_chunks) > 0 and isinstance(W_out_chunks[0], torch.Tensor) and W_out_chunks[0].dim() == 3:
                # W_out_chunks may be a list of tensors per chunk (if using functorch) or per-sample
                if W_out_chunks[0].shape[0] == (end - start):
                    # chunks were appended as tensors
                    W_out = torch.cat(W_out_chunks, dim=0).to(z.device)
                else:
                    # appended per-sample tensors
                    W_out = torch.stack(W_out_chunks, dim=0).to(z.device)
            else:
                W_out = torch.stack(W_out_chunks, dim=0).to(z.device)

            if self.output_dist == 'gaussian':
                return distribution.DiagonalGaussian(mu_flat, self.logvar), W_out
            elif self.output_dist == 'bernoulli':
                mu = torch.sigmoid(mu_flat)
                mu_clip = torch.clamp(mu, min=1e-5, max=1.0 - 1e-5)
                self.logvar = -torch.log((mu_clip * (1 - mu_clip)) + 1e-5)
                return distribution.Bernoulli(mu), W_out
            else:
                raise ValueError
        else:
            if self.output_dist == 'gaussian':
                return distribution.DiagonalGaussian(mu_flat, self.logvar)
            elif self.output_dist == 'bernoulli':
                mu = torch.sigmoid(mu_flat)
                return distribution.Bernoulli(mu)
            else:
                raise ValueError
