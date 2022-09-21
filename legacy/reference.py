#%%
import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
import openpnm.models as mods

Lc = 40.5e-6
# 1. Set up network
sgl = op.network.Cubic(shape=[26, 26, 10], spacing=Lc, name='SGL10BA')
sgl.add_boundary_pores()
proj = sgl.project
wrk = op.Workspace()
wrk.settings['loglevel'] = 50
# 2. Set up geometries
Ps = sgl.pores('*boundary', mode='not')
Ts = sgl.find_neighbor_throats(pores=Ps, mode='xnor', flatten=True)
geo = op.geometry.GenericGeometry(network=sgl,pores=Ps,throats=Ts,name='geo')
geo.add_model(propname='pore.diameter',
              model=mods.geometry.pore_size.weibull,
              shape=3.07,
              loc=19.97e-6,
              scale=1.6e-5)
geo.add_model(propname='throat.diameter',
              model=mods.geometry.throat_size.weibull,
              shape=3.07,
              loc=19.97e-6,
              scale=1.6e-5)
geo.add_model(propname='pore.area',
              model=mods.geometry.pore_area.sphere)
geo.add_model(propname='pore.volume',
              model=mods.geometry.pore_volume.sphere)
geo.add_model(propname='throat.length',
              model=mods.geometry.throat_length.ctc)
geo.add_model(propname='throat.volume',
              model=mods.geometry.throat_volume.cylinder)
geo.add_model(propname='throat.area',
              model=mods.geometry.throat_area.cylinder)
geo.add_model(propname='throat.surface_area',
              model=mods.geometry.throat_surface_area.cylinder)
geo.add_model(propname='throat.endpoints',
              model=mods.geometry.throat_endpoints.spherical_pores)
geo.add_model(propname='throat.conduit_lengths',
              model=mods.geometry.throat_length.conduit_lengths)
Ps = sgl.pores('*boundary')
Ts = sgl.find_neighbor_throats(pores=Ps, mode='or')
boun = op.geometry.Boundary(network=sgl, pores=Ps, throats=Ts, name='boun')

throats = geo.throats()
connected_pores = sgl.find_connected_pores(throats)
x1 = [sgl['pore.coords'][pair[0]][0] for pair in connected_pores]
x2 = [sgl['pore.coords'][pair[1]][0] for pair in connected_pores]
same_x = [x - y == 0 for x, y in zip(x1,x2)]
factor = [s*.95 + (not s)*1 for s in same_x]
throat_diameters = sgl['throat.diameter'][throats]*factor
geo['throat.diameter'] = throat_diameters
geo.regenerate_models(exclude=['throat.diameter'])

air = op.phases.Air(network = sgl, name = 'air')
water = op.phases.Water(network = sgl, name = 'water')
# Reset pore contact angle
water['pore.contact_angle'] = 100.0

phys_water = op.physics.Standard(network=sgl, phase=water, geometry=geo)
phys_air = op.physics.Standard(network=sgl, phase=air, geometry=geo)
phys_water_b = op.physics.Standard(network=sgl, phase=water, geometry=boun)
phys_air_b = op.physics.Standard(network=sgl, phase=air, geometry=boun)

inlets = sgl.pores('bottom_boundary')
used_inlets = [inlets[x] for x in range(0, len(inlets), 2)]

OP_1 = op.algorithms.OrdinaryPercolation(project=proj)
OP_1.set_inlets(pores=used_inlets)
OP_1.setup(phase=water, pore_volume='pore.volume', throat_volume='throat.volume')
OP_1.run(points=100)


data = OP_1.get_intrusion_data()
# Filter for evenly spaced sat inc. first and last
filter_pc = [data.Pcap[0]]
sat = [data.Snwp[0]]
for i, pc in enumerate(data.Pcap):
    if  data.Snwp[i] - sat[-1] > 0.05:
        filter_pc.append(pc)
        sat.append(data.Snwp[i])
filter_pc.append(data.Pcap[-1])
sat.append(data.Snwp[-1])



def update_phase_and_phys(results):
    water['pore.occupancy'] = results['pore.occupancy']
    air['pore.occupancy'] = 1-results['pore.occupancy']
    water['throat.occupancy'] = results['throat.occupancy']
    air['throat.occupancy'] = 1-results['throat.occupancy']
    # Add multiphase conductances
    mode='strict'
    phys_air.add_model(model=mods.physics.multiphase.conduit_conductance,
                       propname='throat.conduit_diffusive_conductance',
                       throat_conductance='throat.diffusive_conductance',
                       mode=mode)
    phys_water.add_model(model=mods.physics.multiphase.conduit_conductance,
                         propname='throat.conduit_diffusive_conductance',
                         throat_conductance='throat.diffusive_conductance',
                         mode=mode)
    phys_air.add_model(model=mods.physics.multiphase.conduit_conductance,
                       propname='throat.conduit_hydraulic_conductance',
                       throat_conductance='throat.hydraulic_conductance',
                       mode=mode)
    phys_water.add_model(model=mods.physics.multiphase.conduit_conductance,
                         propname='throat.conduit_hydraulic_conductance',
                         throat_conductance='throat.hydraulic_conductance',
                         mode=mode)
    phys_air_b.add_model(model=mods.physics.multiphase.conduit_conductance,
                         propname='throat.conduit_diffusive_conductance',
                         throat_conductance='throat.diffusive_conductance',
                         mode=mode)
    phys_water_b.add_model(model=mods.physics.multiphase.conduit_conductance,
                           propname='throat.conduit_diffusive_conductance',
                           throat_conductance='throat.diffusive_conductance',
                           mode=mode)
    phys_air_b.add_model(model=mods.physics.multiphase.conduit_conductance,
                         propname='throat.conduit_hydraulic_conductance',
                         throat_conductance='throat.hydraulic_conductance',
                         mode=mode)
    phys_water_b.add_model(model=mods.physics.multiphase.conduit_conductance,
                           propname='throat.conduit_hydraulic_conductance',
                           throat_conductance='throat.hydraulic_conductance',
                           mode=mode)


perm_air = {'0': [], '1': [], '2': []}
diff_air = {'0': [], '1': [], '2': []}
perm_water = {'0': [], '1': [], '2': []}
diff_water = {'0': [], '1': [], '2': []}

max_Pc = max(OP_1['throat.invasion_pressure'])

num_seq = 20
pore_volumes = sgl['pore.volume']
throat_volumes = sgl['throat.volume']
totV = np.sum(pore_volumes) + np.sum(throat_volumes)

K_air_single_phase = [None, None, None]
D_air_single_phase = [None, None, None]
K_water_single_phase = [None, None, None]
D_water_single_phase = [None, None, None]
bounds = [['front', 'back'], ['left', 'right'], ['top', 'bottom']]

for bound_increment in range(len(bounds)):
    # Run Single phase algs effective properties
    BC1_pores = sgl.pores(labels=bounds[bound_increment][0]+'_boundary')
    BC2_pores = sgl.pores(labels=bounds[bound_increment][1]+'_boundary')
    
    # Effective permeability : air
    sf_air = op.algorithms.StokesFlow(network=sgl, phase=air)
    sf_air.setup(conductance='throat.hydraulic_conductance')
    sf_air.set_value_BC(values=0.6, pores=BC1_pores)
    sf_air.set_value_BC(values=0.2, pores=BC2_pores)
    sf_air.run()
    K_air_single_phase[bound_increment] = sf_air.calc_effective_permeability()
    proj.purge_object(obj=sf_air)
    
    # Effective diffusivity : air
    fd_air = op.algorithms.FickianDiffusion(network=sgl,phase=air)
    fd_air.setup(conductance='throat.diffusive_conductance')
    fd_air.set_value_BC(values=0.6, pores=BC1_pores)
    fd_air.set_value_BC(values=0.2, pores=BC2_pores)
    fd_air.run()
    D_air_single_phase[bound_increment] = fd_air.calc_effective_diffusivity()
    proj.purge_object(obj=fd_air)
    
    # Effective permeability : water
    sf_water = op.algorithms.StokesFlow(network=sgl, phase=water)
    sf_water.setup(conductance='throat.hydraulic_conductance')
    sf_water.set_value_BC(values=0.6, pores=BC1_pores)
    sf_water.set_value_BC(values=0.2, pores=BC2_pores)
    sf_water.run()
    K_water_single_phase[bound_increment] = sf_water.calc_effective_permeability()
    proj.purge_object(obj=sf_water)
    
    # Effective diffusivity : water
    fd_water = op.algorithms.FickianDiffusion(network=sgl,phase=water)
    fd_water.setup(conductance='throat.diffusive_conductance')
    fd_water.set_value_BC(values=0.6, pores=BC1_pores)
    fd_water.set_value_BC(values=0.2, pores=BC2_pores)
    fd_water.run()
    D_water_single_phase[bound_increment] = fd_water.calc_effective_diffusivity()
    proj.purge_object(obj=fd_water)


for Pc in filter_pc:
    update_phase_and_phys(OP_1.results(Pc=Pc))
    print('-' * 80)
    print('Pc', Pc)
    for bound_increment in range(len(bounds)):
        BC1_pores = sgl.pores(labels=bounds[bound_increment][0]+'_boundary')
        BC2_pores = sgl.pores(labels=bounds[bound_increment][1]+'_boundary')

        # Multiphase
        sf_air = op.algorithms.StokesFlow(network=sgl,phase=air)
        sf_air.setup(conductance='throat.conduit_hydraulic_conductance')
        sf_water = op.algorithms.StokesFlow(network=sgl,phase=water)
        sf_water.setup(conductance='throat.conduit_hydraulic_conductance')

        fd_air = op.algorithms.FickianDiffusion(network=sgl,phase=air)
        fd_air.setup(conductance='throat.conduit_diffusive_conductance')
        fd_water = op.algorithms.FickianDiffusion(network=sgl,phase=water)
        fd_water.setup(conductance='throat.conduit_diffusive_conductance')

        #BC1
        sf_air.set_value_BC(values=0.6, pores=BC1_pores)
        sf_water.set_value_BC(values=0.6, pores=BC1_pores)
        fd_air.set_value_BC(values=0.6, pores=BC1_pores)
        fd_water.set_value_BC(values=0.6, pores=BC1_pores)

        #BC2
        sf_air.set_value_BC(values=0.2, pores=BC2_pores)
        sf_water.set_value_BC(values=0.2, pores=BC2_pores)
        fd_air.set_value_BC(values=0.2, pores=BC2_pores)
        fd_water.set_value_BC(values=0.2, pores=BC2_pores)

        # Run Multiphase algs
        sf_air.run()
        sf_water.run()
        fd_air.run()
        fd_water.run()

        Keff_air_mphase = sf_air.calc_effective_permeability()
        Deff_air_mphase = fd_air.calc_effective_diffusivity()
        Keff_water_mphase = sf_air.calc_effective_permeability()
        Deff_water_mphase = fd_water.calc_effective_diffusivity()

        Kr_eff_air = Keff_air_mphase / K_air_single_phase[bound_increment]
        Kr_eff_water = Keff_water_mphase / K_water_single_phase[bound_increment]
        Dr_eff_air = Deff_air_mphase / D_air_single_phase[bound_increment]
        Dr_eff_water = Deff_water_mphase / D_water_single_phase[bound_increment]

        perm_air[str(bound_increment)].append(Kr_eff_air)
        diff_air[str(bound_increment)].append(Dr_eff_air)
        perm_water[str(bound_increment)].append(Kr_eff_water)
        diff_water[str(bound_increment)].append(Dr_eff_water)
        
        proj.purge_object(obj=sf_air)
        proj.purge_object(obj=sf_water)
        proj.purge_object(obj=fd_air)
        proj.purge_object(obj=fd_water)


#%%
#%%
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import euler_number
import porespy as ps
import matplotlib.pyplot as plt
import os
import numpy as np
import openpnm as op
import openpnm.models as mods
from openpnm.models import physics as pm
import pandas as pd

np.random.seed(0)

def n2_cal(pore_index,pore_occ):
    # cal of redundant loop
    return (pore_occ[pore_index[0]] & 
            pore_occ[pore_index[1]])

def n3_cal(pore_index,pore_occ):
    # cal of isolated pore
    return not (pore_occ[pore_index[0]] | 
            pore_occ[pore_index[1]])

# PATH = r'C:\Users\rtopa\OneDrive\phd22\gan\pnm\openpnm'
# PATH = r'C:\Users\rtopa\OneDrive\phd22\pnm\openpnm'
# os.chdir(PATH)
im = np.load('1.npy')
# raw_file = np.fromfile('Berea_2d25um_binary.raw', dtype=np.uint8)
# im = (raw_file.reshape(1000,1000,1000))

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

# define phase and physics
air = op.phases.Air(network=pn)
water = op.phases.Water(network=pn)
phys_air = op.physics.Standard(network=pn, phase=air, geometry=geo)
phys_water = op.physics.Standard(network=pn, phase=water, geometry=geo)
#%% ordinary percolation
# https://notebook.community/PMEAL/OpenPNM/examples/paper_recreations/Gostick%20et%20al.%20(2007)/Gostick%20et%20al.%20(2007)
inlets = pn.pores('bottom')

OP_1 = op.algorithms.OrdinaryPercolation(network=pn,phase=air)
OP_1.set_inlets(pores=inlets)
OP_1.setup(phase=water, pore_volume='pore.volume', throat_volume='throat.volume')
OP_1.run(points=100)

data = OP_1.get_intrusion_data()
sw = [1-x for x in data.Snwp]
f = plt.figure()
plt.plot(sw,data.Pcap)
plt.xlabel('Wetting phase saturation')
plt.ylabel('Capilary pressure')


#%%
filter_pc = [data.Pcap[0]]
sat = [data.Snwp[0]]
for i, pc in enumerate(data.Pcap):
    if  data.Snwp[i] - sat[-1] > 0.01:
        filter_pc.append(pc)
        sat.append(data.Snwp[i])
filter_pc.append(data.Pcap[-1])
sat.append(data.Snwp[-1])
sw = [1-x for x in sat]

f = plt.figure()
plt.plot(sw,filter_pc)
plt.xlabel('Wetting phase saturation')
plt.ylabel('Capilary pressure')

#%%
# demonstrate the invasion phase occupancy
results = OP_1.results(Pc=1e03)
print(results.keys())


# %%
def update_phase_and_phys(results):
    water['pore.occupancy'] = 1-results['pore.occupancy']
    air['pore.occupancy'] = results['pore.occupancy']
    water['throat.occupancy'] = 1-results['throat.occupancy']
    air['throat.occupancy'] = results['throat.occupancy']
    # Add multiphase conductances
    mode='strict'
    # Determines the conductance of a pore-throat-pore conduit 
    # based on the invaded state of each element.


    phys_air.add_model(model=mods.physics.multiphase.conduit_conductance,
                       propname='throat.conduit_hydraulic_conductance',
                       throat_conductance='throat.hydraulic_conductance',
                       mode=mode)
    phys_water.add_model(model=mods.physics.multiphase.conduit_conductance,
                         propname='throat.conduit_hydraulic_conductance',
                         throat_conductance='throat.hydraulic_conductance',
                         mode=mode)


update_phase_and_phys(OP_1.results(Pc=1e3))


# %% absolute permeability calculation
inlet = pn.pores('bottom')
outlet = pn.pores('top')
sf_air = op.algorithms.StokesFlow(network=pn, phase=air)
sf_air.setup(conductance='throat.hydraulic_conductance')
sf_air.set_value_BC(values=0.6, pores=inlet)
sf_air.set_value_BC(values=0.2, pores=outlet)
sf_air.run()
K_air_single_phase = sf_air.calc_effective_permeability()
proj.purge_object(obj=sf_air)


sf_water = op.algorithms.StokesFlow(network=pn, phase=water)
sf_water.setup(conductance='throat.hydraulic_conductance')
sf_water.set_value_BC(values=0.6, pores=inlet)
sf_water.set_value_BC(values=0.2, pores=outlet)
sf_water.run()
K_water_single_phase = sf_water.calc_effective_permeability()
proj.purge_object(obj=sf_water)


# %%
perm_air = []
perm_water = []

for Pc in filter_pc:
    update_phase_and_phys(OP_1.results(Pc=Pc))
    print('-'*80)
    print('Pc',Pc)

    # set up flow simulator
    sf_air = op.algorithms.StokesFlow(network=pn,phase=air)
    sf_air.setup(conductance='throat.conduit_hydraulic_conductance')
    sf_water = op.algorithms.StokesFlow(network=pn,phase=water)
    sf_water.setup(conductance='throat.conduit_hydraulic_conductance')

    # set up boundary conditions
    sf_air.set_value_BC(values=0.6, pores=inlet)
    sf_water.set_value_BC(values=0.6, pores=inlet)
    sf_air.set_value_BC(values=0.2, pores=outlet)
    sf_water.set_value_BC(values=0.2, pores=outlet)

    sf_air.run()
    sf_water.run()

    Keff_air_mphase = sf_air.calc_effective_permeability()
    Keff_water_mphase = sf_air.calc_effective_permeability()

    # cal of kr
    Kr_eff_air = Keff_air_mphase / K_air_single_phase
    Kr_eff_water = Keff_water_mphase / K_water_single_phase

    perm_air.append(Kr_eff_air)
    perm_water.append(Kr_eff_water)
    proj.purge_object(obj=sf_air)
    proj.purge_object(obj=sf_water)


# %%
plt.plot(sat,perm_air)
# %%
