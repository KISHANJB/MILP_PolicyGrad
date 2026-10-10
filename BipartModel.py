"""PyTorch port of learn2branch's GCNN baseline (models/baseline/model.py).

Given a bipartite MILP state (constraints, edges, variables), GCNPolicy
returns one real-valued score per variable. It knows nothing about strong
branching or any other labeling rule -- how those scores get trained
(classification against a single expert pick, regression against a score,
a ranking loss, ...) is entirely a training-script concern.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNormException(Exception):
    # print("Pass pre norm Exception")
    pass


class PreNormLayer(nn.Module):
    """Affine (shift, scale) layer whose parameters are *not* learned by
    gradient descent: call start_updates(), run some batches through the
    model (forward() will raise PreNormException each time, after
    accumulating running mean/var), then stop_updates() to freeze it."""

    def __init__(self, n_units, shift=True, scale=True):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False

    def forward(self, x):
        if self.waiting_updates:
            self._update_stats(x)
            self.received_updates = True
            raise PreNormException
        if self.shift is not None:
            x = x + self.shift
        if self.scale is not None:
            # print("Performing shifting and scaling in PreNormLayer")
            x = x * self.scale
        return x

    def start_updates(self):
        # print ("Start updates in PreNormLayer")
        self.avg, self.var, self.count = 0.0, 0.0, 0.0
        self.waiting_updates, self.received_updates = True, False

    @torch.no_grad()
    def _update_stats(self, x):
        # print("Calculating stats for PreNormLayer")
        x = x.reshape(-1, self.n_units).float()
        sample_avg = x.mean(dim=0)
        sample_var = ((x - sample_avg) ** 2).mean(dim=0)
        sample_count = x.numel() / self.n_units

        delta = sample_avg - self.avg
        m2 = (self.var * self.count + sample_var * sample_count
              + delta ** 2 * self.count * sample_count / (self.count + sample_count))

        self.count += sample_count
        self.avg = self.avg + delta * sample_count / self.count
        self.var = m2 / self.count if self.count > 0 else 1.0

    def stop_updates(self):
        # print("Stop updates PreNormLayer")
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg
        if self.scale is not None:
            var = torch.where(self.var == 0, torch.ones_like(self.var), self.var)
            self.scale = 1 / var.sqrt()
        del self.avg, self.var, self.count
        self.waiting_updates = False


class BipartiteGraphConvolution(nn.Module):
    """One half-convolution (eq. 4): messages are formed per-edge from the
    two endpoint embeddings + edge features, then summed onto one side."""

    def __init__(self, emb_size, edge_nfeats, right_to_left=False):
        super().__init__()
        self.emb_size = emb_size
        self.right_to_left = right_to_left

        self.feature_module_left = nn.Linear(emb_size, emb_size)
        self.feature_module_edge = nn.Linear(edge_nfeats, emb_size, bias=False) # bias is set false becuase for
        self.feature_module_right = nn.Linear(emb_size, emb_size, bias=False)   # linearly added layers, only one bias is sufficent
        self.feature_module_final = nn.Sequential(
            PreNormLayer(1, shift=False),  # single scalar rescale after the sum
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = PreNormLayer(1, shift=False)  # ditto, after the scatter-sum
        self.output_module_BiG = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features, scatter_size):
        # print("Forward Bi GC")
        src, dst = edge_indices[0], edge_indices[1]

        joint = (self.feature_module_left(left_features)[src]
                 + self.feature_module_edge(edge_features)
                 + self.feature_module_right(right_features)[dst])
        joint = self.feature_module_final(joint)

        idx = src if self.right_to_left else dst
        out = joint.new_zeros(scatter_size, self.emb_size).index_add_(0, idx, joint)
        out = self.post_conv_module(out)

        prev = left_features if self.right_to_left else right_features
        return self.output_module_BiG(torch.cat([out, prev], dim=1))


class GCNPolicy(nn.Module):
    def __init__(self, cons_nfeats=5, edge_nfeats=1, var_nfeats=19, emb_size=64):
    # def __init__(self, cons_nfeats=1, edge_nfeats=1, var_nfeats=1, emb_size=4):
        # print("Before GCN init")
        super().__init__()
        # print("After GCN init")

        def embed(n_feats):
            # print("Embedding ",n_feats," features.")
            return nn.Sequential(
                PreNormLayer(n_feats),
                nn.Linear(n_feats, emb_size), nn.ReLU(),
                nn.Linear(emb_size, emb_size), nn.ReLU(),
            )

        self.cons_embedding = embed(cons_nfeats)
        self.edge_embedding = PreNormLayer(edge_nfeats)
        self.var_embedding = embed(var_nfeats)

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size, edge_nfeats, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size, edge_nfeats)

        self.output_module = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )

        # print("Length of self.modules: ",len(list(self.modules)))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        # print("Forward GCNN")
        c = self.cons_embedding(constraint_features)
        e = self.edge_embedding(edge_features)
        v = self.var_embedding(variable_features)

        c = F.relu(self.conv_v_to_c(c, edge_indices, e, v, c.size(0)))
        v = F.relu(self.conv_c_to_v(c, edge_indices, e, v, v.size(0)))

        return self.output_module(v).squeeze(-1)  # one score per variable

    # --- pretraining (calibrates every PreNormLayer, one at a time, in
    # forward-execution order -- see the exception trick in PreNormLayer) ---
    def pre_train_init(self):
        # print("Pre-Train GCNN")
        for m in self.modules():
            if isinstance(m, PreNormLayer):
                m.start_updates()

    def pre_train_next(self):
        # print("Pre-Train-Next GCNN")
        for m in self.modules():
            if isinstance(m, PreNormLayer) and m.waiting_updates and m.received_updates:
                # print("Going to stop updates")
                m.stop_updates()
                return m
        return None