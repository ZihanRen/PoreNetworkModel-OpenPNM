# %%
import numpy as np
import openpnm as op
import porespy as ps
import matplotlib.pyplot as plt
import pandas as pd
import os
from openpnm.models import physics as mods
from scipy import stats
from skimage.measure import euler_number

np.random.seed(10)
np.set_printoptions(precision=4)
resolution = 2.32e-06
im_shape = 128
snwp = np.arange(0, 1.1, 0.1)


def phi_cal(geo,size,resolution):
    v_tot = (size*resolution)**3
    vol_pores = geo['pore.volume'].sum()
    vol_throats = geo['throat.volume'].sum()
    phi = (vol_pores+vol_throats)/v_tot
    return phi

def invasion(phase,inlets):
    OP_1 = op.algorithms.OrdinaryPercolation(network=pn,phase=phase)
    OP_1.set_inlets(pores=inlets)
    OP_1.setup(phase=air, pore_volume='pore.volume', throat_volume='throat.volume')
    OP_1.run(points=100)
    return OP_1

def network_extract(im,resolution):

    snow = ps.networks.snow(
    im=im,
    voxel_size=resolution)

    proj = op.io.PoreSpy.import_data(snow)

    return proj

def update_phase_and_phys_water(results):
    # convert results to boolean matrix
    results['pore.occupancy'] = results['pore.occupancy'].astype(int) > 0
    results['throat.occupancy'] = results['throat.occupancy'].astype(int) > 0
    val = np.amin(phys_water['throat.hydraulic_conductance'])/1000

    # results: pore and throats occupancy
    air.update(results)
    # if stokes flow of water, we need to update phys_water hydraulic conductance.
    #  The percoaltion result is always refering to air since this is phase we we are using to invade
    # Therefore, locations where water hydraulic conductance is slow indicate gas saturated
    phys_water['throat.hydraulic_conductance'][air['throat.occupancy']] = val

def update_phase_and_phys_air(results):
    val = np.amin(phys_water['throat.hydraulic_conductance'])/1000

    results['pore.occupancy'] = results['pore.occupancy'].astype(int) > 0
    results['throat.occupancy'] = results['throat.occupancy'].astype(int) > 0
    # results: pore and throats occupancy
    air.update(results)
    # if stokes flow of air, we need to update phys_air hydraulic conductance.
    # Therefore, locations where air hydraulic conductance is slow indicate water saturated
    phys_air['throat.hydraulic_conductance'][~air['throat.occupancy']] = val


def load_nparray(fname,name):
    # fname: image index
    # name: rock type
    # PATH: current working directory
    load_PATH = name+'-sub/'+ fname
    im1 = np.load(load_PATH)
    return im1


def cal_absolute_perm(image_size,resolution,Q):
    A = (image_size*image_size) *resolution**2 # m^2
    L = image_size * resolution # m
    mu = water['pore.viscosity'].max() # N s/m^2 or Pa s
    delta_P = 1 # 1-0
    K = Q * L * mu / (A * delta_P)
    return K/0.98e-12*1000

def cal_eff_perm(image_size,resolution,Q,phase):
    A = (image_size*image_size) *resolution**2
    L = image_size * resolution
    mu = phase['pore.viscosity'].max() # N s/m^2 or Pa s
    delta_P = 1
    K = Q * L * mu / (A * delta_P)

    return K/0.98e-12*1000



#%%
im = np.load('1.npy')
ws = op.Workspace()
resolution = 2.25e-6 
snow = ps.networks.snow(
    im=im,
    voxel_size=resolution)

proj = op.io.PoreSpy.import_data(snow)
pn,geo = proj[0],proj[1]
data_tmp = {}

data_tmp['coordination'] = np.mean(pn.num_neighbors(pn.Ps))
data_tmp['porosity'] = ps.metrics.porosity(im)
data_tmp['pore.diameter'] = np.mean(geo['pore.diameter'])
data_tmp['throat.diameter'] = np.mean(geo['throat.diameter'])
data_tmp['euler'] = euler_number(im,connectivity=3)

health = pn.check_network_health()
op.topotools.trim(network=pn, pores=health['trim_pores'])
# define phase and physics
air = op.phases.Air(network=pn)
water = op.phases.Water(network=pn)
water['pore.contact_angle'] = 0
air['pore.contact_angle'] = 180

phys_air = op.physics.Standard(network=pn, phase=air, geometry=geo)
phys_water = op.physics.Standard(network=pn, phase=water, geometry=geo)

inlets = pn.pores('bottom')
outlets = pn.pores('top')

# perform invasion flow simulation
OP_1=invasion(air,inlets)
data = OP_1.get_intrusion_data()
sw = [1-x for x in data.Snwp]
snw = data.Snwp

# %% stoke flow simulate absolute perm
phys_water.regenerate_models()
st = op.algorithms.StokesFlow(network=pn)
st.setup(phase=water)
st.set_value_BC(pores=inlets, values=0)
st.set_value_BC(pores=outlets, values=1) # this part needs some validation
st.run()
Q_abs_water = st.rate(pores=outlets,mode='group')
kabs_tmp = cal_absolute_perm(im_shape,resolution,Q_abs_water[0])
phys_water.regenerate_models()

# simulation of water
kw_tmp = []
phys_water.regenerate_models()
for Pc in data.Pcap:
    update_phase_and_phys_water(OP_1.results(Pc=Pc))
    st = op.algorithms.StokesFlow(network=pn)
    st.setup(phase=water)
    st.set_value_BC(pores=inlets, values=0)
    st.set_value_BC(pores=outlets, values=1)
    st.run()
    Q = st.rate(pores=outlets,mode='group')[0]
    keff = cal_eff_perm(im_shape,resolution,Q,water)
    kw_tmp.append(keff)
    proj.purge_object(obj=st)
    phys_water.regenerate_models()

# simulation of air
knw_tmp = []
phys_air.regenerate_models()
for Pc in data.Pcap:
    update_phase_and_phys_air(OP_1.results(Pc=Pc))
    st = op.algorithms.StokesFlow(network=pn)
    st.setup(phase=air)
    st.set_value_BC(pores=inlets, values=0)
    st.set_value_BC(pores=outlets, values=1)
    st.run()
    Q = st.rate(pores=outlets,mode='group')[0]
    keff = cal_eff_perm(im_shape,resolution,Q,air)
    knw_tmp.append(keff)
    proj.purge_object(obj=st)
    phys_air.regenerate_models()

# calcualte kr
krw_tmp = np.array(kw_tmp)/kabs_tmp
krnw_tmp = np.array(knw_tmp)/kabs_tmp

data_tmp['kr_water'] = krw_tmp
data_tmp['kr_air'] = krnw_tmp
data_tmp['k_water'] = np.array(kw_tmp)
data_tmp['k_air'] = np.array(krnw_tmp)
data_tmp['kabs'] = kabs_tmp
data_tmp['snw'] = np.array(snw)
data_tmp['sw'] = np.array(sw)
# data[fname] = data_tmp
# ws.close_project(proj)


#%% save dictionary file
# os.chdir(r'C:\Users\rtopa\OneDrive\phd22\gan\3d\data-gen')
os.chdir('/akshat/s0/zur74/OneDrive/phd22/gan/3d/data-gen')
import pickle
with open("features-3d-1.pkl", "wb") as tf:
pickle.dump(data,tf)

# load dictionary file
# import pickle
# with open("test.pkl", "rb") as tf:
#     new_dict = pickle.load(tf)


