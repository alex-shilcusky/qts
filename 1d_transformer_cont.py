import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence
from jax._src import dtypes
import netket as nk
import time
import json
import matplotlib.pyplot as pl
import os
from netket import nn as nknn
import numpy as np

print('\n#################################################')


class Transformer(nn.Module):
    L: int # chain length
    b: int # cluster length
    h: int 
    stdev: float
    
    @nn.compact
    def __call__(self, x):
        # print('\n x = ',x )
        x = jnp.atleast_2d(x)
        return jax.vmap(self.evaluate_single, in_axes=(0))(x)
        
    def evaluate_single(self, x):
        Nc = self.L // self.b
        x = x.reshape(Nc, self.b)

        Q = self.param('Q', nn.initializers.normal(stddev=stdev), (self.b * self.h, self.b), jnp.complex128)
        K = self.param('K', nn.initializers.normal(stddev=stdev), (self.b * self.h, self.b), jnp.complex128)
        V = self.param('V', nn.initializers.normal(stddev=stdev), (self.b * self.h, self.b), jnp.complex128)
        # W = self.param('W', nn.initializers.normal(stddev=stdev), (self.L, self.L), jnp.complex128)
        W = nn.Dense(features=self.L,
                     use_bias=True,
                     param_dtype=jnp.complex128,
                     kernel_init=nn.initializers.normal(stddev=stdev),
                     bias_init=nn.initializers.normal(stddev=stdev))

    
        Qx = jnp.matmul(x, Q.T).reshape(Nc, self.h, self.b)
        Kx = jnp.matmul(x, K.T).reshape(Nc, self.h, self.b)
        Vx = jnp.matmul(x, V.T).reshape(Nc, self.h, self.b) # (L//b, h, b)

        # def compute_attention(i):
        #     z = jnp.matmul(Qx[:, i, :], Kx[:, i, :].T) / jnp.sqrt(self.b)
        #     # alist = nk.nn.softmax(-jnp.diag(z))
        #     alist = nk.nn.softmax(-z)
        #     alist = jnp.diag(alist)
        #     return (alist[:, jnp.newaxis] * Vx[:, i, :])
        def compute_attention(i):
            z = jnp.matmul(Qx[:, i, :], Kx[:, i, :].T) / jnp.sqrt(self.b)
            z = nk.nn.softmax(-z)
            return z @ Vx[:, i, :]
        
        vtilde = jax.vmap(compute_attention, in_axes=(0))(jnp.arange(self.h))
        vtilde = vtilde.reshape(self.h, Nc*self.b) # (h, L)
        return jnp.sum(vtilde * W(vtilde))
    






L = 8
b = 4
h = 2
J2 = 0
# seed = 0

lattice = nk.graph.Chain(length=L, pbc=True, max_neighbor_order=2)
hilbert = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J = [1.0, J2], sign_rule=[False, False]) 

# lattice, hilbert, hamiltonian will be global variables

print('\n Lattice is bipartite? ')
print(lattice.is_bipartite())

if 0: # if we want to compute the exact GS energy, or just take it from known value
    if (L <= 16):
        evals = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=False)
        exact_gs_energy = evals[0]
    else:
        exact_gs_energy = -0.4438 * L * 4

    if L==40 and J2==0:
        exact_gs_energy = -0.443663 * L * 4
    if L == 100 and J2==0:
     exact_gs_energy = -0.4432295 * L * 4

    print('The exact ground-state energy is E0 = ', exact_gs_energy)
    print('Ground state energy per site = ',exact_gs_energy/L/4 )



def initialize_vstate(L, b, h, N_samples, stdev, seed):
    wf = Transformer(L, b, h, stdev)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, num=2)
    params = wf.init(subkey, jnp.zeros((1,lattice.n_nodes)))
    init_samples = jnp.zeros((1,))
    key, subkey = jax.random.split(key, 2)
    sampler_seed = subkey
    sampler = nk.sampler.MetropolisExchange(hilbert=hilbert,
                                            graph=lattice,
                                            d_max=L)
    vstate = nk.vqs.MCState(sampler=sampler, 
                            model=wf, 
                            sampler_seed=sampler_seed,
                            n_samples=N_samples)
    return vstate


def optimize_vstate(vstate, N_opt, fname, eta, diag_shift, seed):

    optimizer = nk.optimizer.Sgd(learning_rate=eta)
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=True) # need small diag_shift for Transformer 
    vmc = nk.VMC(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        preconditioner=sr,
        variational_state=vstate)
    
    callback = nk.callbacks.InvalidLossStopping(monitor='mean', patience=1)
    start = time.time()
    vmc.run(out = 'Transformer', n_iter = N_opt, callback=callback)
    end = time.time()
    print('Runtime: ', end-start)

    data = json.load(open("Transformer.log"))
    iters = data["Energy"]["iters"]
    energy_Re = data["Energy"]["Mean"]["real"]
    energy_Im = data["Energy"]["Mean"]["imag"]
    var = data['Energy']['Variance']
    sig = data['Energy']['Sigma']
    R_hat = data['Energy']['R_hat']
    TauCorr = data['Energy']['TauCorr']

    # close("Transformer.log")
    if 1: # for saving EE, iters
        num = 0
        folder_name = 'data_'+fname
        path = os.path.join('continuous_runs', folder_name)
        filename = 'seed=%i.npz'%seed

        if os.path.exists(os.path.join(path, filename)):
            print(' !!! Found old file !!!')
            prev_data = np.load(os.path.join(path, filename), allow_pickle=True)
            prev_iters = prev_data['iters']
            prev_Er = prev_data['energy_Re']
            prev_Ei = prev_data['energy_Im']
            prev_var = prev_data['var']
            prev_sig = prev_data['sig']
            prev_Rhat = prev_data['R_hat']
            prev_TauCorr = prev_data['TauCorr']

            # prev
            iters = list(prev_iters) + [max(prev_iters)+1 + iter for iter in iters]
            energy_Re = list(prev_Er) + energy_Re 
            energy_Im = list(prev_Ei) + energy_Im
            var = list(prev_var) + var
            sig = list(prev_sig) + sig
            R_hat = list(prev_Rhat) + R_hat
            TauCorr = list(prev_TauCorr) + TauCorr

            # want to configure all data as lists to make saving 'clean'

            os.remove(os.path.join(path, filename))
            print('\n Removed old file')

        np.savez(os.path.join(path, filename),
            iters=iters,
            energy_Re=energy_Re,
            energy_Im=energy_Im,
            var=var,
            sig=sig,
            R_hat=R_hat,
            TauCorr=TauCorr
            )

        

    if 1: # for saving vstate
        folder_name = 'vstates_'+fname
        vstate_path = os.path.join('continuous_runs', folder_name)
        vstate_fname = 'vstate_seed=%i'%seed
        if os.path.exists(os.path.join(vstate_path, vstate_fname)):
            os.remove(os.path.join(vstate_path, vstate_fname))

        with open(os.path.join(vstate_path, vstate_fname), 'wb') as file:
            file.write(flax.serialization.to_bytes(vstate))
    return 

diag_shift = 0.001
eta = 0.01
N_opt = 1
N_samples = 1008 # number monte carlo samples

seed = 0
stdev = 0.1
vstate = initialize_vstate(L,b,h, N_samples, stdev, seed)
print('Number of parameters = ', nk.jax.tree_size(vstate.parameters))


fname = f'L={L}_b={b}_h={h}_J2={J2}_eta={eta}_stdev={stdev}' # folder name
folder_name = 'data_'+fname
path = os.path.join('continuous_runs', folder_name)
if not os.path.exists(path): # 
    print('MAKING DIRECTORY!')
    os.makedirs(path)

folder_name = 'vstates_'+fname
vstate_path = os.path.join('continuous_runs', folder_name)
if not os.path.exists(vstate_path):
    print('MAKING DIRECTORY!')
    os.makedirs(vstate_path)

N_opt = 10

start = time.time()
for i in range(4):
    optimize_vstate(vstate, N_opt, fname, eta, diag_shift, seed)
# optimize_vstate(vstate, 100, fname, eta, diag_shift, seed)
end = time.time()
print('\n Time to do save/load shenanigans for 100 steps')
print(end-start, ' sec')

filename = 'seed=%i.npz'%seed
filename = os.path.join(path, filename)


data = np.load(filename, allow_pickle=True)

print('\n ### \n')
print('does file exist?')
print(os.path.exists(filename))
# print(data)
# print(type(data))
# data = dict(data)
EE = data['energy_Re']
iters = data['iters']
print(iters)
print(EE)

pl.figure()
pl.plot(iters,EE)
pl.show()

# print(EE)
# print(type(EE))




# fname = os.path.join('continuous_runs', fname)
# data = np.load(fname)
# # for file in os.listdir('continuous_runs'):
# #     print(file)

# print(data)

# optimize_vstate(vstate0, 10, eta, diag_shift)

