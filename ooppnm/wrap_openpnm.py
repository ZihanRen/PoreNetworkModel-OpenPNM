import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
import random
import pandas as pd


class Wrap_pnm:

    def __init__(self,PATH=None,im=None,voxel_size=2.32e-06,inlet='left',outlet='right'):

        # read numpy image PATH 
        if im is not None:
            self.im = im
        else:
            self.im = np.load(PATH)
        if ( (PATH==None) and (im==None) ):
            raise Exception("Sorry you need to provide both image PATH or image array")

        self.ws = op.Workspace()
        self.voxel_size = voxel_size
        self.inlet_dir = inlet
        self.outlet_dir = outlet
        
        # initialize OpenPNM objects
        self.pn = None
        self.air = None
        self.water = None
        self.ip = None


    def network_extract(self):
        snow = ps.networks.snow2(
            phases=self.im,
            phase_alias={True:"void",False:"solid"},
            voxel_size=self.voxel_size)
        
        self.pn = op.io.network_from_porespy(snow.network)
        self.pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        self.pn.regenerate_models()
        

    def add_boundary_pn(self):
        
        # add boundary pores
        min_coor = self.pn['pore.coords'].min()
        max_coor = self.pn['pore.coords'].max()

        def marker_arr(min_value,max_value,boundary='left'):

            if boundary == 'left':
                marker = [
                        min_value,
                        random.uniform(min_value,max_value),
                        random.uniform(min_value,max_value)
                        ]

            if boundary == "right":
                marker = [
                max_value,
                random.uniform(min_value,max_value),
                random.uniform(min_value,max_value)
                ]
            
            return marker

        inlet_marker = [marker_arr(min_coor,max_coor,self.inlet_dir) for _ in range(20)]
        outlet_marker = [marker_arr(min_coor,max_coor,self.outlet_dir) for _ in range(20)]
        inlet_marker = np.array(inlet_marker)
        outlet_marker = np.array(outlet_marker)

        op.topotools.find_surface_pores(network=self.pn, markers=inlet_marker, label=self.inlet_dir)
        op.topotools.find_surface_pores(network=self.pn, markers=outlet_marker, label=self.outlet_dir)

    
    def init_physics(self):
        # should return self.air and physics
        self.air = op.phase.Air(network=self.pn,name='self.air')
        self.air['pore.surface_tension'] = 0.072
        self.air['pore.contact_angle'] = 180.0
        self.air.add_model_collection(op.models.collections.phase.air)
        self.air.add_model_collection(op.models.collections.physics.basic)
        self.air.regenerate_models()

        self.water = op.phase.Water(network=self.pn,name='water')
        self.water.add_model_collection(op.models.collections.phase.water)
        self.water.add_model_collection(op.models.collections.physics.basic)
        self.water.regenerate_models()


    def invasion_percolation(self):
        # should return invasion percolation object
        self.ip = op.algorithms.InvasionPercolation(network=self.pn, phase=self.air)
        Finlets_init = self.pn.pores(self.inlet_dir)
        Finlets=([Finlets_init[x] for x in range(0, len(Finlets_init), 2)])
        self.ip.set_inlet_BC(pores=Finlets)
        self.ip.run()


    def __sat_occ_update(self,network, nwp, wp, ip, i):
        pore_mask = ip['pore.invasion_sequence'] < i
        throat_mask = ip['throat.invasion_sequence'] < i
        sat_p = np.sum(network['pore.volume'][pore_mask])
        sat_t = np.sum(network['throat.volume'][throat_mask])
        sat1 = sat_p + sat_t
        bulk = network['pore.volume'].sum() + network['throat.volume'].sum()
        sat = sat1/bulk
        nwp['pore.occupancy'] = pore_mask
        nwp['throat.occupancy'] = throat_mask
        wp['throat.occupancy'] = 1-throat_mask
        wp['pore.occupancy'] = 1-pore_mask
        return sat

    def __Rate_calc(self,network, phase, inlet, outlet, conductance):
        phase.regenerate_models()
        St_p = op.algorithms.StokesFlow(network=network, phase=phase)
        St_p.settings._update({'conductance' : conductance})
        St_p.set_value_BC(pores=inlet, values=1)
        St_p.set_value_BC(pores=outlet, values=0)
        St_p.run()
        val = np.abs(St_p.rate(pores=inlet, mode='group'))
        return val

    def kr_simulation(self,Snwp_num=100):
    
        flow_in = self.pn.pores(self.inlet_dir)
        flow_out = self.pn.pores(self.outlet_dir)
        model_mp_cond = op.models.physics.multiphase.conduit_conductance
        self.air.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                    throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
        self.water.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                    throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')

        max_seq = np.max([np.max(self.ip['pore.invasion_sequence']),
                np.max(self.ip['throat.invasion_sequence'])])
        
        start = 0
        stop = max_seq
        step = max_seq//Snwp_num
        Snwparr = []
        relperm_nwp = []
        relperm_wp = []

        for i in range(start, stop, step):
            self.air.regenerate_models()
            self.water.regenerate_models()
            sat = self.__sat_occ_update(network=self.pn, nwp=self.air, wp=self.water, ip=self.ip, i=i)
            Snwparr.append(sat)
            Rate_abs_nwp = self.__Rate_calc(self.pn, self.air, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
            Rate_abs_wp = self.__Rate_calc(self.pn, self.water, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
            Rate_enwp = self.__Rate_calc(self.pn, self.air, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
            Rate_ewp = self.__Rate_calc(self.pn, self.water, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
            relperm_nwp.append(Rate_enwp/Rate_abs_nwp)
            relperm_wp.append(Rate_ewp/Rate_abs_wp)

        # output pandas dataframe
        kr_data = {
            'snwp':Snwparr,
            'krnw':np.array(relperm_nwp).flatten(),
            'krw':np.array(relperm_wp).flatten()
            }

        kr_data_df = pd.DataFrame(kr_data)

        return kr_data_df

    def close_ws(self):
        self.ws.close_project()

    def crop_kr(self,df):
        df_crop = df[ (df['krw']>0.03) ]
        return df_crop

    def kr_visualize(self,df):
        f = plt.figure(figsize=[8,8])
        plt.plot(df['snwp'], df['krnw'], '*-', label='Kr_nw')
        plt.plot(df['snwp'], df['krw'], 'o-', label='Kr_w')
        plt.xlabel('Snwp')
        plt.xlim([0,1])
        plt.ylabel('Kr')
        plt.title('Relative Permeability drainage curve')
        plt.legend()




        
    

