
# coding: utf-8

# ## Wang-Landau Sampling of lattice model

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Energy_calculation as EC
import Monte_carlo as MC
import copy
import time


# In[2]:


#


# In[9]:


class Lattice:
    def __init__(self,lat_sz=2):
        ''' Class to initialize lattice
                
        '''
        
        # Characteristics:
        self.lattice = np.zeros((lat_sz,lat_sz))
        self.num_pos_ions = 12
        self.num_polymer = 6
        self.pos_ion_coords = []
        self.polymer_coords = []
        self.water_coords = []
        self.water_charge = 0.1
        self.reject = 0 # Number of MC rejects
        self.temperature = 1
    def charge_up(self):
        ''' Add charges to lattice, don't add to sites already occupied
        '''
        (xspan,yspan) = self.lattice.shape
        
        # Place polymer 
        conformer = 3 # 3 -- 1 is straight, 2 is 90 deg bend, 3 is extended
        coordinates = np.load("6_length_BME.npy").item()
        polymer_coords = coordinates[conformer]
        polymer_coords = [[coord[0],coord[1]] for coord in polymer_coords]
        for coord in polymer_coords:
            self.lattice[coord[0],coord[1]] = -1
            self.polymer_coords.append(coord)
            
        flag = False # To place randomly
        while flag:
            r1,r2 = np.random.randint(0,xspan,2)
            if [r1,r2] not in self.polymer_coords:
                self.lattice[r1,r2] = -1
                self.polymer_coords.append([r1,r2])
            if len(self.polymer_coords) == self.num_polymer:
                flag = False
                
                
        # Place ions 
        flag = True
        while flag:
            r1,r2 = np.random.randint(0,xspan,2)
            if [r1,r2] not in self.pos_ion_coords and [r1,r2] not in self.polymer_coords:
                self.lattice[r1,r2] = 1
                self.pos_ion_coords.append([r1,r2])
            if len(self.pos_ion_coords) == self.num_pos_ions:
                flag = False  


                
        # Place Water        
        for idx in range(xspan):
            for idy in range(yspan):
                if [idx,idy] not in self.pos_ion_coords and [idx,idy] not in self.polymer_coords:
                    self.lattice[idx,idy] = self.water_charge
                    self.water_coords.append([idx,idy])
                    
    def get_energy_edges(self):
        """ Evaluate what minimum and maximum energies are for wang-landau sampling and split into bins
            where no bin has a zero count
        """
        energy_store = []
        N = 10000
        ii = 0
        z1 = copy.deepcopy(self)
        while ii < N:
            z2,r1,r2 = MC.IonMove(z1)
            delG = MC.Energy_Difference(z1,z2,r1,r2)
            z2.energy += delG
            energy_store.append(z2.energy)
            z1 = copy.deepcopy(z2)
            del(z2)
            ii += 1

        # Find unique values and then keep reducing bin size until all bins have at least one count
        tmp1 = list(set(energy_store))
        bins = 80
        tmp2 = np.histogram(tmp1,bins=bins)
        while 0 in tmp2[0]:
            bins -= 1
            tmp2 = np.histogram(tmp1,bins=bins)
        print("# of bins: ",bins)
        self.energy_edges = tmp2[1]
        return tmp1


# In[ ]:




    


# In[4]:


def wang_landau(self):
    """ Perform wang_landau simulation
        Arguments:
            self {class} -- Lattice class
    """
    # Possible energies
    energy_edges = self.energy_edges
    energy_edges = energy_edges[:-1]
    energy_store = []
    hist = np.zeros(len(energy_edges))
    entropy = np.zeros(len(energy_edges))
    ln_factor = 1
    ii = 1
    
    while ln_factor > 1e-3 and ii < 80000:
        # Perform Monte Carlo Move on lattice
        z1,r1,r2 = MC.IonMove(self)
        delG = MC.Energy_Difference(self,z1,r1,r2)
        z1.energy += delG
        energy_store.append(z1.energy)
        new_edge = np.digitize(z1.energy,energy_edges) - 1 
        old_edge = np.digitize(self.energy,energy_edges) - 1
        P = np.exp(entropy[old_edge]-entropy[new_edge])
        if P > np.random.rand():
            old_edge = new_edge
            self = copy.deepcopy(z1)
        entropy[old_edge] += ln_factor
        hist[old_edge] += 1
        del(z1)
        
        if is_flat(hist):
            hist[:] = 0
            ln_factor /= 2
            print(ln_factor,ii)
        ii += 1
    print("ln(f): ",ln_factor,"iter #: ",ii)
    return hist, entropy, energy_store

def is_flat(hist):
    """ Check is histogram of counting is flat
    """
    minH = np.min(hist)
    if minH > 0.8*np.mean(hist):
        return True
    return False


# In[10]:


z1 = Lattice(14)
z1.charge_up()
z1.energy = EC.Energy(z1) - EC.energy_polymer(z1)


# In[11]:


# z1.get_energy_edges()

## Water Charge of 0
#z1.energy_edges = np.array([-11.53599972, -11.19739429, -10.85878885, -10.52018342,
#       -10.18157798,  -9.84297255,  -9.50436711,  -9.16576167,
#        -8.82715624,  -8.4885508 ,  -8.14994537,  -7.81133993,
#        -7.4727345 ,  -7.13412906,  -6.79552362,  -6.45691819,
#        -6.11831275,  -5.77970732,  -5.44110188,  -5.10249644,
#        -4.76389101,  -4.42528557,  -4.08668014,  -3.7480747 ,
#        -3.40946927,  -3.07086383,  -2.73225839,  -2.39365296,
#        -2.05504752,  -1.71644209,  -1.37783665,  -1.03923122,
#        -0.70062578,  -0.36202034,  -0.02341491,   0.31519053,
#         0.65379596,   0.9924014 ,   1.33100683,   1.66961227,
#         2.00821771,   2.34682314,   2.68542858,   3.02403401,
#         3.36263945,   3.70124489]) 

## Water Charge of 0.01
#z1.energy_edges = np.array([-11.11644926, -10.81820032, -10.51995138, -10.22170245,
#        -9.92345351,  -9.62520457,  -9.32695563,  -9.02870669,
#        -8.73045775,  -8.43220881,  -8.13395987,  -7.83571093,
#        -7.53746199,  -7.23921305,  -6.94096411,  -6.64271517,
#        -6.34446623,  -6.0462173 ,  -5.74796836,  -5.44971942,
#        -5.15147048,  -4.85322154,  -4.5549726 ,  -4.25672366,
#        -3.95847472,  -3.66022578,  -3.36197684,  -3.0637279 ,
#        -2.76547896,  -2.46723002,  -2.16898108,  -1.87073215,
#        -1.57248321,  -1.27423427,  -0.97598533,  -0.67773639,
#        -0.37948745,  -0.08123851,   0.21701043,   0.51525937,
#         0.81350831,   1.11175725,   1.41000619])


## Water Charge of 0.1 for 6 ions
#z1.energy_edges = np.array([-1.77589559, -1.59714658, -1.41839758, -1.23964858, -1.06089958,
#       -0.88215058, -0.70340158, -0.52465258, -0.34590358, -0.16715458,
#        0.01159442,  0.19034342,  0.36909242,  0.54784142,  0.72659042,
#        0.90533943,  1.08408843,  1.26283743,  1.44158643,  1.62033543,
#        1.79908443,  1.97783343,  2.15658243,  2.33533143,  2.51408043,
#        2.69282943,  2.87157843,  3.05032743,  3.22907643,  3.40782544,
#        3.58657444,  3.76532344,  3.94407244,  4.12282144,  4.30157044,
#        4.48031944,  4.65906844,  4.83781744,  5.01656644,  5.19531544,
#        5.37406444,  5.55281344,  5.73156244,  5.91031145,  6.08906045,
#        6.26780945,  6.44655845,  6.62530745,  6.80405645,  6.98280545,
#        7.16155445,  7.34030345,  7.51905245,  7.69780145,  7.87655045,
#        8.05529945,  8.23404845,  8.41279746,  8.59154646,  8.77029546,
#        8.94904446,  9.12779346,  9.30654246,  9.48529146,  9.66404046,
#        9.84278946, 10.02153846, 10.20028746, 10.37903646, 10.55778546])

## Water Charge of 0.1 for 12 ions
z1.energy_edges = np.array([ 3.75812243,  4.20849443,  4.65886643,  5.10923843,  5.55961043,
        6.00998243,  6.46035443,  6.91072643,  7.36109843,  7.81147043,
        8.26184243,  8.71221443,  9.16258643,  9.61295843, 10.06333043,
       10.51370243, 10.96407443, 11.41444643, 11.86481843, 12.31519043,
       12.76556243, 13.21593443, 13.66630643, 14.11667843, 14.56705043,
       15.01742243, 15.46779443, 15.91816643, 16.36853843, 16.81891043,
       17.26928243, 17.71965443, 18.17002643, 18.62039843, 19.07077043,
       19.52114243, 19.97151443, 20.42188642, 20.87225842, 21.32263042,
       21.77300242, 22.22337442, 22.67374642, 23.12411842])

# In[7]:


start = time.time()

hist, entropy, energy_store = wang_landau(z1)
end = time.time()
print("Elapsed Time: ",end-start)


# In[ ]:


np.savetxt("entropy_12_3.txt",entropy)
np.savetxt("histogram_12_3.txt",hist)
np.savetxt("energy_store_12_3.txt",energy_store)
np.savetxt("energy_edges_12_3.txt",z1.energy_edges)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




