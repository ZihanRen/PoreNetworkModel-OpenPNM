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

# PATH = r'C:\Users\rtopa\OneDrive\phd22\gan\pnm\openpnm'
# PATH = r'C:\Users\rtopa\OneDrive\phd22\pnm\openpnm'
# os.chdir(PATH)
im = np.load('1.npy')
# raw_file = np.fromfile('Berea_2d25um_binary.raw', dtype=np.uint8)
# im = (raw_file.reshape(1000,1000,1000))

# running snow algorithm to extract pore geometry information from network
snow = ps.networks.snow(
    im,
    voxel_size=2.32e-06)

# extracting network and geometry from snow simulation results
proj = op.io.PoreSpy.import_data(snow)

pn,geo = proj[0],proj[1]

# check the health of network
health = pn.check_network_health()
op.topotools.trim(network=pn, pores=health['trim_pores'])


# %%
import openpnm.models.physics as pmods
air = op.phases.Air(network=pn)
water = op.phases.Water(network=pn)
water['pore.contact_angle'] = 0
air['pore.contact_angle'] = 180
air['pore.surface_tension'] = 0.072
air.regenerate_models()
# water['pore.surface_tension'] = 0.064
# air['pore.surface_tension'] = 0.064
# phys_air = op.physics.Standard(network=pn, phase=air, geometry=geo)
# phys_water = op.physics.Standard(network=pn, phase=water, geometry=geo)


air = op.phases.Air(network=pn,name='air')
water = op.phases.Water(network=pn,name='water')
air.add_model(propname='throat.hydraulic_conductance',
              model=pmods.hydraulic_conductance.hagen_poiseuille)
air.add_model(propname='throat.entry_pressure',
              model=pmods.capillary_pressure.washburn)
water.add_model(propname='throat.hydraulic_conductance',
                model=pmods.hydraulic_conductance.hagen_poiseuille)
water.add_model(propname='throat.entry_pressure',
                model=pmods.capillary_pressure.washburn)

#%%
ip = op.algorithms.InvasionPercolation(network=pn, phase=air)
Finlets_init = pn.pores('left')
Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init), 2)])
ip.set_inlets(pores=Finlets)
ip.run()



# %%
air.update(ip.results())
rp = op.algorithms.metrics.RelativePermeability(network=pn)
rp.settings.update({'nwp': 'air',
                    'invasion_sequence': 'invasion_sequence'})
rp.run(Snwp_num=10)




# %%
results=rp.get_Kr_data()
pd.DataFrame(results['kr_nwp'])
fig = rp.plot_Kr_curves()