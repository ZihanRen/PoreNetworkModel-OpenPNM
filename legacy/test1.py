#%%
import numpy as np
import pandas as pd
import openpnm as op
import openpnm.models.physics as pmods
import openpnm.models.geometry as pmods_geo

import matplotlib.pyplot as plt
pn = op.network.Cubic(shape=[15,15,15], spacing=6e-5)
geom = op.geometry.SpheresAndCylinders(network=pn, pores=pn['pore.all'],
                                throats=pn['throat.all'])

geom.add_model(
    propname='pore.area',
    model=pmods_geo.pore_area.sphere)

geom.add_model(
    propname='throat.area',
    model=pmods_geo.throat_area.cylinder)

geom.add_model(
    propname='throat.conduit_lengths.pore1',
    model=pmods_geo.conduit_lengths.spheres_and_cylinders)

geom.add_model(
    propname='throat.conduit_lengths.throat',
    model=pmods_geo.conduit_lengths.spheres_and_cylinders)
geom.regenerate_models()

#%%
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
Finlets= pn.pores('top')
ip.set_inlets(pores=Finlets)
ip.run()
air.update(ip.results())

#%%
rp = op.algorithms.metrics.RelativePermeability(network=pn)
rp.setup(
    invading_phase='air',
    defending_phase='water',
    invasion_sequence='invasion_sequence',
    )

# rp.run(Snwp_num=10)


#%%
rp.settings.update({'nwp': 'air',
                    'invasion_sequence': 'invasion_sequence'})
rp.run(Snwp_num=10)

results=rp.get_Kr_data()
pd.DataFrame(results['kr_nwp'])

fig = rp.plot_Kr_curves()

# %%
