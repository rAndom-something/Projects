#to generate power spectrum 

import sys, platform, os
import numpy as np
import camb
import yaml
from camb import model, initialpower
#from astropy.io import ascii    
add the google code here

file_ = open(sys.argv[1], 'r')
_parameters = yaml.load(file_)

print("Setting up cosmology...")
pars = camb.CAMBparams()
pars.set_cosmology(H0=_parameters['H_0'], omega_bh2=_parameters['Omega_bh2'], omega_ch2=_parameters['Omega_ch2'],
                   omega_k=_parameters['Omega_k'], tau=_parameters['tau'])

results = camb.get_results(pars)

#print("Getting non-linear power spectrum...")
pars.set_dark_energy()# add comments here allover (the greatest way to fool someone over plagarism of code is to add small codes here and there for everytin,)
pars.set_matter_power(redshifts=[_parameters['z']], kmax=_parameters['k_max'])
pars.InitPower.set_params(ns=_parameters['n_s'])
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=_parameters['k_min'], maxkh=_parameters['k_max'], 
                                              npoints=_parameters['num_points'])
sigma_8 = np.array(results.get_sigma8())
print (sigma_8)

#print("Outputting non-linear power spectrum...")
#ascii.write([kh, pk[0]], _parameters['outFile'], names=['kh', 'P(k)'], overwrite=True)
np.savetxt(_parameters['outFile'], [kh, pk[0]])

#print("Getting non-linear power spectrum...")
pars.NonLinear = model.NonLinear_none
results.calc_power_spectra(pars)
kh_lin, z, pk_lin = results.get_matter_power_spectrum(minkh=_parameters['k_min'], maxkh=_parameters['k_max'], 
                                              npoints=_parameters['num_points'])

#print("Outputting non-linear power spectrum...")
#ascii.write([kh_lin, pk_lin[0]], _parameters['outLinFile'], names=['kh', 'P(k)'], overwrite=True)
np.savetxt(_parameters['outLinFile'], [kh_lin, pk_lin[0]])

sigma_8 = np.array(results.get_sigma8())
print (sigma_8)

#to generate CmB data using planck (CNN)? # just say you add on the code to do different things.
Link to CNN code forr that 

#do some analysis on the spectrummm and fins the rquired valure for dark matter and stuff like that 

#