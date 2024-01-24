"""
Bit packing. Each agent is represented by 32 bits. This example uses 22 bits.
"""

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from enum import Enum
from methods import *

# our disease states
class States(Enum):
    NULL = 0
    S = 1
    I = 2
    R = 3

if __name__ == '__main__':
        
    # setup the way we will store our info
    attr_sizes = OrderedDict()
    attr_sizes['node']  = 5  # node id
    attr_sizes['age']   = 15 # in days
    attr_sizes['state'] = 2  # Null = 0,S = 1,I = 2, R = 3

    # create the bit mapping (attribute: [ start, size])
    bm = OrderedDict()
    cnt = 0
    for k,v in attr_sizes.items():
        bm[k] = [cnt, v]
        cnt += cnt + v
        
    print('max size for our attributes:')
    for k,v in attr_sizes.items():
        print(k, 2**v + 1)
    print('\n bit mapping (starting bit position, number of bits):')
    for k,v in bm.items():
        print(k, v)


    # Single node simulation
    num_agents = 2_000_000
    seed_infections = 10
    R0 = 2.5
    gamma = 1/10
    num_timesteps = 365

    # initialize the agent array
    agents = np.zeros(num_agents, dtype=np.uint32)

    # put everyone in node 1 for fun
    agents = set_bits(agents, 1, *bm['node'])

    # let's initialize everyone to susceptible
    agents = set_bits(agents, States.S.value, *bm['state'])

    # seed infections
    seeds = np.random.randint(0, num_agents, seed_infections)
    agents[seeds] = set_bits(agents[seeds],  States.I.value, *bm['state'])
    print('Number of agents infected:', np.sum(check_bits(agents, *bm['state'], States.I.value)))

    count = lambda state: np.sum(check_bits(agents, *bm['state'], state))

    # main loop
    record = np.zeros((num_timesteps, 4)) # t, S, I, R
    for t in tqdm(range(num_timesteps)):

        # record
        record[t,:] = np.array([t, count(States.S.value), count(States.I.value), count(States.R.value)])
        
        # calculate the force of infection
        foi = np.sum(check_bits(agents, *bm['state'], States.I.value)) * R0 * gamma / num_agents

        # recover
        mask = np.logical_and(np.random.rand(num_agents) < gamma, check_bits(agents, *bm['state'], States.I.value))
        agents[mask] = set_bits(agents[mask], States.R.value, *bm['state'])
        
        # infect 
        mask = np.logical_and(np.random.rand(num_agents) < foi, check_bits(agents, *bm['state'], States.S.value))
        agents[mask] = set_bits(agents[mask], States.I.value, *bm['state'])
        
    
    np.savetxt('record.csv', record, delimiter=',', fmt='%d')
    print('done')