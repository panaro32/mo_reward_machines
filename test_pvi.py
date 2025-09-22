from reward_machines import RewardMachine, RewardMachineEnv
from labeled_envs import GridWorldEnv
from vi import vi, vi_rm
from pvi import pvi, pvi_rm

rm1 = RewardMachine('rm_files/abb_once.rm')
rm2 = RewardMachine('rm_files/baa_cycle.rm')
rm  = RewardMachine(['rm_files/abb_once.rm', 'rm_files/baa_cycle.rm'])
env = GridWorldEnv()
rm1_env = RewardMachineEnv(env, rm1)
rm2_env = RewardMachineEnv(env, rm2)
rm_env  = RewardMachineEnv(env, rm)

u10, u20, u00 = rm1.initial_state, rm2.initial_state, rm.initial_state
s00 = tuple(env.agent_pos)

print(f'RM1 = {vi_rm(rm1)[u10]}')
print(f'RM2 = {vi_rm(rm2)[u20]}')
print(f'RM1_ENV = {vi(rm1_env)[(s00, u10)]}')
print(f'RM2_ENV = {vi(rm2_env)[(s00, u20)]}')

print(f'RM1 = {pvi_rm(rm1)[u10]}')
print(f'RM2 = {pvi_rm(rm2)[u20]}')
print(f'RM1_ENV = {pvi(rm1_env)[(s00, u10)]}')
print(f'RM2_ENV = {pvi(rm2_env)[(s00, u20)]}')

import matplotlib.pyplot as plt

pareto_rm = pvi_rm(rm)[u00]
print(f'RM = {pareto_rm}')
obj1_rm, obj2_rm = zip(*pareto_rm)
plt.scatter(x=obj1_rm, y=obj2_rm)
plt.show()

pareto = pvi(rm_env)[(s00, u00)]
print(f'RM_ENV = {pareto}')
obj1, obj2 = zip(*pareto)
plt.scatter(x=obj1, y=obj2)
plt.show()
