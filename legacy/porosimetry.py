#%%
from PIL import Image
import os
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import euler_number
import matplotlib.pyplot as plt
import os
import numpy as np
import openpnm as op
import openpnm.models as mods
from openpnm.models import physics as pm
import pandas as pd

np.random.seed(0)
im = np.load('1.npy')


# running snow algorithm to extract pore geometry information from network
snow = ps.networks.snow(
    im=im,
    voxel_size=2.32e-06)

# extracting network and geometry from snow simulation results
proj = op.io.PoreSpy.import_data(snow)

pn,geo = proj[0],proj[1]

# check the health of network
health = pn.check_network_health()
op.topotools.trim(network=pn, pores=health['trim_pores'])

air = op.phases.Air(network=pn)
water = op.phases.Water(network=pn)
water['pore.contact_angle'] = 0
air['pore.contact_angle'] = 180


phys_air = op.physics.Standard(network=pn, phase=air, geometry=geo)
phys_water = op.physics.Standard(network=pn, phase=water, geometry=geo)
# %%
mip = op.algorithms.Porosimetry(network=pn, phase=air)
mip.set_inlets(pores=pn.pores('front'))
mip.run(100)

# %%
Pc, Snwp = mip.get_intrusion_data()
fig, ax = plt.subplots()
ax.semilogx(Snwp,Pc, 'r*-')
ax.set_ylabel("Capillary pressure (Pa)")
ax.set_xlabel("Saturation")
# %%
plt.scatter(Snwp,Pc)
# %%
