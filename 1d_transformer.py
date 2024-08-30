from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence
from jax._src import dtypes
import netket as nk
import time
import json
import matplotlib.pyplot as plt
import os

print('#################################################')

class Transformer(nn.Module):
    L: int # chain length
    b: int # cluster length
    h: int 

    
    @nn.compact
    def __call__(self, x):
        # print('\n x = ',x )
        x = jnp.atleast_2d(x)
        return jax.vmap(self.evaluate_single, in_axes=(0))(x)
        
    def evaluate_single(self, x):
        Nc = self.L // self.b
        x = x.reshape(Nc, self.b)

        stdev = 0.1
        Q = self.param('Q', nn.initializers.normal(stddev=stdev), (self.b * self.h, self.b), jnp.complex128)
        K = self.param('K', nn.initializers.normal(stddev=stdev), (self.b * self.h, self.b), jnp.complex128)
        V = self.param('V', nn.initializers.normal(stddev=stdev), (self.b * self.h, self.b), jnp.complex128)
        # W = self.param('W', nn.initializers.normal(stddev=stdev), (self.L, self.L), jnp.complex128)
        W = nn.Dense(features=self.L,
                     use_bias=True,
                     param_dtype=jnp.complex128,
                     kernel_init=nn.initializers.normal(stddev=0.1),
                     bias_init=nn.initializers.normal(stddev=0.1))

    
        Qx = jnp.matmul(x, Q.T).reshape(Nc, self.h, self.b)
        Kx = jnp.matmul(x, K.T).reshape(Nc, self.h, self.b)
        Vx = jnp.matmul(x, V.T).reshape(Nc, self.h, self.b)

        def compute_attention(i):
            z = jnp.matmul(Qx[:, i, :], Kx[:, i, :].T) / jnp.sqrt(self.b)
            # alist = nk.nn.softmax(-jnp.diag(z))
            alist = nk.nn.softmax(-z)
            alist = jnp.diag(alist)
            return (alist[:, jnp.newaxis] * Vx[:, i, :])
        
        vtilde = jax.vmap(compute_attention, in_axes=(0))(jnp.arange(self.h))
        vtilde = vtilde.reshape(self.h, Nc*self.b) # (h, L)
        return jnp.sum(vtilde * W(vtilde))

L = 8
b = 4
h = 2

# diag_shift = 0.001
diag_shift = 0.001
eta = 0.01
N_opt = 100
# N_samples = 1008 # number monte carlo samples
N_samples = 3024
N_discard = 0

J2 = 0

lattice = nk.graph.Chain(length=L, pbc=True, max_neighbor_order=2)
# lattice = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J = [1.0, J2], sign_rule=[False, False]) 


from netket.models import RBM, RBMSymm
from netket.operator.spin import sigmax, sigmaz, sigmay 
import numpy as np


print('\n Lattice is bipartite? ')
print(lattice.is_bipartite())

if (L <= 16):
    evals = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=False)
    exact_gs_energy = evals[0]
else:
    exact_gs_energy = -0.4438 * L * 4

if L==40 and J2==0:
    exact_gs_energy = -0.443663 * L * 4
    
print('The exact ground-state energy is E0 = ', exact_gs_energy)
print('Ground state energy per site = ',exact_gs_energy/L/4 )

wf = Transformer(L, b, h)

######################################################################
if 1: # for psuedorandom seed
    sampler = nk.sampler.MetropolisExchange(hilbert=hilbert,
                                                graph=lattice)

    vstate = nk.vqs.MCState(sampler=sampler, 
                            model=wf, 
                            n_samples=N_samples)
######################################################################
if 0: #for specifying random seed
    seed = 0
    # seed = 20240823
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, num=2)
    params = wf.init(subkey, jnp.zeros((1,lattice.n_nodes)))
    init_samples = jnp.zeros((1,))
    key, subkey = jax.random.split(key, 2)
    sampler_seed = subkey

    sampler = nk.sampler.MetropolisExchange(hilbert=hilbert,
                                            graph=lattice,
                                            d_max=L,
                                            n_chains=N_samples,
                                            sweep_size=lattice.n_nodes)
    vstate = nk.vqs.MCState(sampler=sampler, 
                            model=wf, 
                            sampler_seed=sampler_seed,
                            n_samples=N_samples, 
                            n_discard_per_chain=N_discard,
                            variables=params)
######################################################################
print('Number of parameters = ', nk.jax.tree_size(vstate.parameters))

if 1:

    optimizer = nk.optimizer.Sgd(learning_rate=eta)
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=True) # need small diag_shift for Transformer 

    vmc = nk.VMC(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        preconditioner=sr,
        variational_state=vstate)

    print(vstate.sampler_state)
    start = time.time()

    # vmc.run(out = 'Transformer', n_iter = N_opt, callback=callback)
    vmc.run(out = 'Transformer', n_iter = N_opt)
    end = time.time()
    print(vstate.sampler_state)
    print('Runtime: ', end-start)


    # print('params: \n', vstate.parameters)

    # import the data from log file
    data = json.load(open("Transformer.log"))
    # Extract the relevant information
    iters = data["Energy"]["iters"]
    energy_Re = data["Energy"]["Mean"]["real"]
    params = vstate.parameters
    # print('\n params = \n', params)

    E = energy_Re[-1]
    print('E = ', E)

    print(abs((E-exact_gs_energy)/exact_gs_energy)*100, ' % error')

    if 0: # for saving
        num = 0
        filename = f'Nopt={N_opt}_Nmc={N_samples}_eta={eta}_num={num}'
        path = f'data_j2={J2}_L={L}_b={b}_h={h}'

        if not os.path.exists(path):
            os.makedirs(path)


        np.savez(os.path.join(path, filename),
            iters=iters, energy=energy_Re,
            exact_energy=exact_gs_energy
            )

    fig, ax1 = plt.subplots()
    ax1.plot(iters, energy_Re,color='C8', label='Energy (ViT)')

    ax1.set_ylabel('Energy')
    ax1.set_xlabel('Iteration')
    ax1.set_title('L=%i, b=%i, h=%i '%(L,b, h))
    # plt.axis([0,iters[-1],exact_gs_energy-1,exact_gs_energy+1])
    plt.axhline(y=exact_gs_energy, xmin=0, xmax=iters[-1], linewidth=2, color='k', label='Exact')

    ax1.legend()
    plt.show()
