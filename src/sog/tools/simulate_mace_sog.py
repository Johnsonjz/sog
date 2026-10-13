from sog import Sog
import torch
sog_arguments={"use_atomwise":False}
m=Sog(sog_arguments=sog_arguments, r_cut=sog_arguments.get("r_cut", None))
r=torch.rand(12,3,dtype=torch.float64)
q=torch.rand(12,dtype=torch.float64)-0.5
q=q-q.mean()
cell=torch.eye(3,dtype=torch.float64).unsqueeze(0)*10.0
out=m(positions=r, cell=cell, latent_charges=q, batch=None, compute_energy=True, compute_bec=False)
print(f"Output keys: {out.keys()}")
if 'energy' in out:
    print(f"E_lr shape: {out['energy'].shape}")
elif 'e_lr' in out:
    print(f"e_lr shape: {out['e_lr'].shape}")
