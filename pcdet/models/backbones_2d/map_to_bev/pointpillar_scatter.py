import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        ## mlp
        # self.mlps = nn.ModuleList()
        # shared_mlps = []
        # mlp_spec = [1,64,1]
        # for k in range(len(mlp_spec)-1):
        #     shared_mlps.extend([
        #         nn.Conv2d(mlp_spec[k], mlp_spec[k+1],
        #                 kernel_size=1, bias=False),
        #         nn.BatchNorm2d(mlp_spec[k+1]),
        #         nn.ReLU()
        #     ])
        # self.mlps = nn.ModuleList().append(nn.Sequential(*shared_mlps))
        # mlp end

        # mlp
        # self.mlps = nn.ModuleList()
        # shared_mlps = []
        # shared_mlps.extend([nn.Conv2d(64, 64,kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.ReLU()])
        # self.mlps = nn.ModuleList().append(nn.Sequential(*shared_mlps))
        # mlp end


        # topk
        self.topk = nn.Linear(64, 64, bias=True)
        self.nm = nn.BatchNorm1d(64)
        self.rl = nn.ReLU()
        self.topk_score = nn.Linear(64, 1, bias=True)
        self.nm_score = nn.BatchNorm1d(1)
        self.rl_score = nn.ReLU()
        # topk



    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :] # 坐标
            flag_mask = this_coords[:,4] != -1
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # torch.Size([8032])
            indices = indices.type(torch.long)
            # print(pillar_features.shape)
            pillars = pillar_features[batch_mask, :]
            pillars = self.topk(pillars)
            pillars = self.nm(pillars)
            pillars = self.rl(pillars)

            score = self.topk_score(pillars)
            score = self.nm_score(score)
            score = self.rl_score(score)  

            # print(pillars.shape)
            pillars = pillars.t()
            indices = indices[flag_mask]
            pillars = pillars[:,flag_mask] # 可能出错
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict