#!/usr/bin/env python

# SJD 19 May 2015
# Script form of gradient descent with momentum.

import argparse
import json
import yaml
import sys
import os
import numpy as np
import pdb

# parse input arguments:
parser = argparse.ArgumentParser(description='SLM optimisation routine using gradient descent + momentum.')
parser.add_argument('params', metavar='parameter_file',
                    help='Parameters file containing gradient descent values, targets, etc.')
args = parser.parse_args()

# put theano after args because import is slow.
import theano

if theano.config.floatX == 'float32' or theano.config.device[0:3] == 'gpu':
    print "Using GPU or float32 precision - aborting.  Run python with THEANO_FLAGS=device=cpu, e.g. 'THEANO_FLAGS=device=cpu,floatX=float64 python script.py'"
    sys.exit(1)

import theano.tensor as T
from slm import slmOptimisation
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Run the code:
    with open(args.params) as f:
        params = yaml.load(f)
        
    # output directory:
    if 'sumatra_label' in params:
        outputdir = os.path.join('data/', params['sumatra_label'])
    else:
        outputdir = 'data/tmp'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    # print header line:
    print json.dumps(params)
    print '--------------------------------'
    print 'cwd: ' + os.getcwd()
    
    #pdb.set_trace()
    
    targetname = os.path.join(params['target'], 'target.dat')
    weightingname = os.path.join(params['target'], params['weighting'], 'weight.dat')
    weightingASname = os.path.join(params['target'], params['weighting'], 'weight_as.dat')

    target = np.loadtxt(targetname)
    weighting = np.loadtxt(weightingname)
    weighting_as = np.loadtxt(weightingASname)

    N = target.shape[0]/2

    # initialise the phase:
    def initial_phase(N):
        """ Return a randomised phase array over [0, 2pi]
        """
        return np.random.uniform(low=0, high=2*np.pi, size=(N,N)).astype(theano.config.floatX)

    init_phi = initial_phase(N)

    # for this example, the profile $S_{nm}$ is uniform.
    s_profile = np.ones_like(init_phi, dtype=complex)  # NB in general s_profile may be complex.

    slmOpt = slmOptimisation(target, init_phi, s_profile, A0=1.0/1000) # fudge factor! Need to stabilise this...
    
    # we now define a cost function to use, squared error for now.
    cost = T.sum(T.pow((slmOpt.E_out_2 - target)*weighting, 4))

    # visualise the output given the initial phi field
    f_E_out = slmOpt.get_E_out()   # these return functions to evaluate the output
    f_E2_out = slmOpt.get_intensity()

    E_out = f_E_out()              # actually calculate the SLM output
    E2_out = f_E2_out()


    l_count = []
    l_cost_SE = []
    l_cost_AS = []
    nn = 0  # global step-counting index

    # new cost function:
    cost_SE   = T.sum(T.pow((slmOpt.E_out_2 - target)*weighting, 2))
    cost_QE   = T.sum(T.pow((slmOpt.E_out_2 - target)*weighting, 4))
    cost_AS_x =  T.sum(T.pow(slmOpt.E_out_2[0:2*N-1,:]-slmOpt.E_out_2[1:2*N,:], 2) * 20*weighting_as[0:2*N-1,:])
    cost_AS_y =  T.sum(T.pow(  slmOpt.E_out_2[:,0:2*N-1]-slmOpt.E_out_2[:,1:2*N], 2) * 20*weighting_as[:,0:2*N-1])

    cost = cost_SE #+ cost_AS_x + cost_AS_y
    grad = T.grad(cost, wrt=slmOpt.phi)

    l_rate = 0.1   # 'learning rate'
    momentum = 0.95 # momentum decay
    updates = ((slmOpt.phi, slmOpt.phi - l_rate * slmOpt.phi_rate),
            (slmOpt.phi_rate, momentum*slmOpt.phi_rate + (1.-momentum)*grad))

    print "Compiling update function..."
    update = theano.function([], 
                            cost, 
                            updates=updates,
                            on_unused_input='warn')
    print "...done"

    C = update()

    f_cost_SE = theano.function([], cost_SE)
    f_cost_AS = theano.function([], cost_AS_x + cost_AS_y)

    f_phi_updates = theano.function([], l_rate*slmOpt.phi_rate)

    #fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10,10))

    # make plots
    E_out = f_E_out()
    E2_out = f_E2_out()

    update_rate_target = 1e-5
    phi_rate_avg = 0.0
    print 'Initial C: {}'.format(C)
    last_C = C
    n = 0
    for n in range(params['gradient_descent']['n_steps']):
        C = update()
        nn += 1
        if np.mod(n, 10) == 0:
            phi_rate_avg += np.mean(np.abs(f_phi_updates())) * 10./250
        if np.mod(n, 1000) == 0:
            filename = os.path.join(outputdir, str(nn) + '.dat')
            np.savetxt(filename, slmOpt.phi.get_value(), fmt='%.2f')
        if np.mod(n, 250) == 0:
            c_SE = float(f_cost_SE())
            c_AS = float(f_cost_AS())
            l_cost_SE.append(c_SE)
            l_cost_AS.append(c_AS)
            l_count.append(nn)
            print '{step:d} Cost (SE):{cost_SE:.2e}   Cost (AS):{cost_AS:.2e}   Steps: mean:{update_step:.2e} max:{max_update_step:.2e}   l_rate:{l_rate:.2e}'.format(
                step=nn,
                cost_SE=c_SE,
                cost_AS=c_AS,
                update_step=np.mean(np.abs(f_phi_updates())),
                max_update_step=np.max(np.abs(f_phi_updates())),
                l_rate=l_rate
            )
            # save the intensity plot:
            # make plots
            E_out = f_E_out()
            E2_out = f_E2_out()
            
            # also renormalise the update rate:
            l_rate = np.min([update_rate_target / phi_rate_avg, 1.5*l_rate])  # can go up by 50% at the most.
            phi_rate_avg = 0.0 # reset
            updates = ((slmOpt.phi, slmOpt.phi - l_rate * slmOpt.phi_rate),
                       (slmOpt.phi_rate, momentum*slmOpt.phi_rate + (1.-momentum)*grad))
            update = theano.function([], 
                                    cost, 
                                    updates=updates,
                                    on_unused_input='warn')

