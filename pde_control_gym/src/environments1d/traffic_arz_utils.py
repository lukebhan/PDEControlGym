
import numpy as np
	

##### Fundamental diagram
def Veq(vm, rm, rho):
	return vm * ( 1 - rho/rm)

# Flux
def F_r(vm, rm, rho, y):
	return y + rho * Veq(vm, rm, rho)

def F_y(vm, rm, rho, y):
	return y * (y/rho + Veq(vm, rm, rho))

# Spatial function
def c_x(self, x): 
	return -1 / self.tau * np.exp(-x/self.tau/self.vs)
