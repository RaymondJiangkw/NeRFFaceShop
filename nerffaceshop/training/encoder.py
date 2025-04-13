import torch
from torch import nn

#----------------------------------------------------------------------------

class MappingNet(torch.nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super().__init__()
        self.layer = layer
        self.first = nn.Linear(coeff_nc, descriptor_nc)

        for i in range(layer):
            net = nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Linear(descriptor_nc, descriptor_nc),
            )
            setattr(self, 'encoder' + str(i), net)

        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder'+ str(i))
            out = model(out) + out
        return out

class Encoder(nn.Module):
    def __init__(self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int, 
        mapping_layers: int, 
        mlp_layers: int, 
    ) -> None:
        super().__init__()
        
        self.mapping_3DMM = MappingNet(
            coeff_nc=in_dim,
            descriptor_nc=hidden_dim,
            layer=mapping_layers,
        )

        self.mapping_Refine = nn.Sequential(
            *[nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1)] * mlp_layers,
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, coeffs):
        feat = self.mapping_3DMM(coeffs)
        feat = self.mapping_Refine[-2:](feat + self.mapping_Refine[:-2](feat))
        return feat