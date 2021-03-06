#!/usr/bin/env python

# 
# 

"""
SJD 19 May 2015
Script form of gradient descent with momentum.

To use, run as:
    python gradient-descent-script.py example.yaml
where example.yaml contains the run parameters.

At the time of writing, example.yaml contained 50000 iterations.  This isn't enough
to achieve a good output intensity pattern, but also will take quite a long time to
run (2 hours on my desktop).  The first 5000 steps are sufficient to see the basic
pattern forming, and vortices starting to move outwards towards the edges.

Has various features:
- plots every (currently) 500 steps
- adjusts step size to prevent update rate slowing
- contains a few possible cost functions
"""

import argparse
import json
import yaml
import sys
import os
import numpy as np

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

# make sure that slm module in on the path:
if '..' not in sys.path:
    sys.path.append('..')

import theano.tensor as T
from slm import slmOptimisation
import pandas as pd
from pandas import DataFrame

import matplotlib
matplotlib.use('Agg')
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
    plotdir = os.path.join(outputdir, 'plots')
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    
    # print header line:
    print json.dumps(params)
    print '--------------------------------'
    
    targetname = os.path.join(params['target'], 'target.dat')
    weightingname = os.path.join(params['target'], params['weighting'], 'weight.dat')
    weightingASname = os.path.join(params['target'], params['weighting'], 'weight_as.dat')

    target = np.loadtxt(targetname)
    weighting = np.loadtxt(weightingname)
    weighting_as = np.loadtxt(weightingASname)

    N = target.shape[0]/2

    plot_args = {'extent':[0, 2*N, 0, 2*N],
                 'interpolation':'None',
                 'origin': 'lower'}
    
    output_line_frequency = params['output_line_frequency']
    output_data_frequency = params['output_data_frequency']
    update_frequency      = params['gradient_descent']['update_frequency']

    if params['initialisation'] is None:
        init_phi = np.random.uniform(low=0, high=2*np.pi, size=(N,N))
    else:
        init_phi = np.loadtxt(params['initialisation'])

    # for this example, the profile $S_{nm}$ is uniform.
    s_profile = np.ones_like(init_phi, dtype=complex)  # NB in general s_profile may be complex.

    slmOpt = slmOptimisation(target, init_phi, s_profile, A0=1.0/1000) # fudge factor! Need to stabilise this...
    
    # visualise the output given the initial phi field
    f_E_out = slmOpt.get_E_out()   # these return functions to evaluate the output
    f_E2_out = slmOpt.get_intensity()

    E_out = f_E_out()              # actually calculate the SLM output
    E2_out = f_E2_out()


    l_count = []
    l_cost_SE = []
    l_cost_QE = []
    l_mean_update = []
    l_max_update = []

    cost_SE   = T.sum(T.pow((slmOpt.E_out_2 - target)*weighting, 2))
    cost_QE   = T.sum(T.pow((slmOpt.E_out_2 - target)*weighting, 4))
    cost_AS_x =  T.sum(T.pow(slmOpt.E_out_2[0:2*N-1,:]-slmOpt.E_out_2[1:2*N,:], 2) * 20*weighting_as[0:2*N-1,:])
    cost_AS_y =  T.sum(T.pow(  slmOpt.E_out_2[:,0:2*N-1]-slmOpt.E_out_2[:,1:2*N], 2) * 20*weighting_as[:,0:2*N-1])

    if params['gradient_descent']['cost'] == 'squared':
        cost = cost_SE
    elif params['gradient_descent']['cost'] == 'quartic':
        cost = cost_QE
    else:
        assert False, 'Need to specify a valid cost function.'
        
    grad = T.grad(cost, wrt=slmOpt.phi)

    l_rate = 0.1   # 'learning rate'
    momentum = params['gradient_descent']['momentum'] # momentum decay
    updates = ((slmOpt.phi, slmOpt.phi - l_rate * slmOpt.phi_rate),
               (slmOpt.phi_rate, momentum*slmOpt.phi_rate + (1.-momentum)*grad))

    print "Compiling update function..."
    update = theano.function([], 
                            cost, 
                            updates=updates,
                            on_unused_input='warn')
    print "...done"

    # take a single step
    C = update()

    f_cost_SE = theano.function([], cost_SE)
    f_cost_QE = theano.function([], cost_QE)
    f_cost_AS = theano.function([], cost_AS_x + cost_AS_y)
    f_phi_updates = theano.function([], l_rate*slmOpt.phi_rate)

    # prepare for plots
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10,10))

    E_out = f_E_out()
    E2_out = f_E2_out()

    update_rate_target = float(params['gradient_descent']['update_rate_target'])
    phi_rate_avg = 0.0
    print 'Initial C: {}'.format(C)
    last_C = C
    n = 0
    for n in xrange(params['gradient_descent']['n_steps']):
        # do update step:
        C = update()
        
        # do various outputs if needed.
        if n % output_data_frequency == 0:
            filename = os.path.join(outputdir, str(n) + '.dat')
            np.savetxt(filename, slmOpt.phi.get_value(), fmt='%.2f')
        if n % output_line_frequency == 0:
            c_SE = float(f_cost_SE())
            c_QE = float(f_cost_QE())
            l_cost_SE.append(c_SE)
            l_cost_QE.append(c_QE)
            l_mean_update.append(np.mean(np.abs(f_phi_updates())))
            l_max_update.append(np.max(f_phi_updates()))
            print '{step:d} Cost (SE):{cost_SE:.2e}   Cost (QE):{cost_QE:.2e}   Steps: mean:{update_step:.2e} max:{max_update_step:.2e}   l_rate:{l_rate:.2e}'.format(
                step=n,
                cost_SE=c_SE,
                cost_QE=c_QE,
                update_step=np.mean(np.abs(f_phi_updates())),
                max_update_step=np.max(np.abs(f_phi_updates())),
                l_rate=l_rate
            )
            # save the intensity plot:
            E_out = f_E_out()
            E2_out = f_E2_out()
            ax.imshow(E2_out[300:400,300:400], vmin=0, vmax=1, **plot_args)
            ax.set_title('Intensity');
            ax2.imshow(E_out[0][300:400,300:400], vmin=-1, vmax=1, **plot_args)
            ax2.set_title('Re(E)');
            fig_name = os.path.join(plotdir, '{n:06d}.png'.format(n=n))
            plt.savefig(fig_name)
            
        if n % update_frequency == 0:
            # also renormalise the update rate:
            phi_rate_avg = np.mean(np.abs(f_phi_updates()))
            l_rate = np.min([update_rate_target / phi_rate_avg, 1.2*l_rate])  # can go up by 20% at the most.
            updates = ((slmOpt.phi, slmOpt.phi - l_rate * slmOpt.phi_rate),
                       (slmOpt.phi_rate, momentum*slmOpt.phi_rate + (1.-momentum)*grad))
            update = theano.function([], 
                                    cost, 
                                    updates=updates,
                                    on_unused_input='warn')
    
    print 'Finished gradient descent, saving summary.'
    # create and save the dataframe with the learning curves:
    df = DataFrame({'Cost_SE': l_cost_SE,
                    'Cost_QE': l_cost_QE,
                    'Mean_update': l_mean_update,
                    'Max_update': l_max_update})
    df.to_pickle(os.path.join(outputdir, 'summary.pkl'))
    
    sys.exit()
