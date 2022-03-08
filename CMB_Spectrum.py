#CMB generator (modified) with added comments of functions
import healpy as hp #healpy is a Python package to handle pixelated data on the sphere
import numpy as np
import matplotlib.pyplot as plt

#we take the Planck 2018 power spectrum and simulate a CMB maps which is a Gaussian realization of this spectrum

cl = hp.read_cl('Cls_Planck2018_lensed_scalar.fits')[0] #inputting data file
l = np.arange(len(cl)) #Return evenly spaced values within a given interval.

plt.loglog(l, cl*l*(l+1)/np.pi/2., lw=4) #formula from theory##
plt.xlim([2.2, 1000]) 
plt.ylim([1e2, 1e4])
plt.xlabel(r'Multipole $\ell$', fontsize=18)
plt.ylabel(r'$C^{TT}_{\ell}\ell(\ell+1)/2\pi$', fontsize=18)
plt.savefig('cmb_ps.pdf')

cmb_map = hp.synfast(cl, nside=2048, new=True) #Create a map(s) from cl


hp.mollview(cmb_map, unit=r'$\mu K$') # mollweide projection with mollview is the most common visualization tool for maps
plt.savefig('CMB_maps.pdf') #save the CMB map

#create the training set for the GAN by taking 1000 flat 5°x5° patches with 64x64 pixels each from the full sky map generated above

patch_dim = 5   #dimension of patches in degrees (patch dimensions)
Npix = 64       #dimension of patches in pixels  (number of pixels within each patch)
reso = patch_dim*Npix/60. #resolution of patches (resolution of the patches)
cmb_patches = []
reso_amin = patch_dim*60./Npix #formula defining parameters
sizepatch = reso_amin/60. ##formula, defining parameters
for N in range(1000): #1000 cause we are going for 100 flat pathches
    lat = np.random.uniform(-90,90) #we can get the random samples from uniform distribution (latitude)
    lon = np.random.uniform(0,360)  #and returns the random samples as numpy array by using this method. (longitude)
    header = set_header(lon, lat, sizepatch, Npix) #setting the header as the given parameters
    cmb_patches.append(h2f(cmb_map, header)) #adding the values to the empty array defined earlier
cmb_patches = np.array(cmb_patches) #creates the given input as an array
window = (cosine_window(Npix)) #Create a window function with a simple cosine shape using number of pixels.

plt.imshow(cmb_patches[2]) #shows the patch created by using data from cmb_patches[2]

#We rescale the patches in the range [-1,1]. This is usefull to achieve a better optimization on the GAN.

cmb_res = np.zeros(np.shape(cmb_patches)) #np.shape(2_arrays_of 4 elements)= (2,4) and adding np.zeros would make 
#an array of zeros containing 2 arrays of 4 zeros 
def rescale_min_max(img, a=-1, b=1, return_min_max=False): #the function rescales the image resolution for a better optimization
    img_resc = (b-a)*(img-np.min(img))/(np.max(img)-np.min(img))+a #resolution formula
    if return_min_max:  #if this true then return min and max of image as well
        return img_resc, np.min(img), np.max(img) 
    else:                #if false
        return img_resc

for i in range(cmb_patches.shape[0]): #rescaling this for every patch
        cmb_res[i] = rescale_min_max(cmb_patches[i])
np.savez('cmb_patches_1000.npz', patches=cmb_res)

#here we calculate the factor needed to rescale back the power spectra of the rescaled maps to the original power
cmb_res = np.load('cmb_patches_1000.npz')['patches']
cl_data_tot = []
for i in range(1000):
    ell, cl_data = calculate_2d_spectrum(cmb_res[i]*window, 30, 2000, reso, 64)
    cl_data_tot.append(cl_data)
bias = cl[ell]/np.mean(cl_data_tot, axis=0)


import os
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D,UpSampling2D, Activation, BatchNormalization
from keras.layers import Reshape, Dense, Input
from keras.layers import LeakyReLU, Dropout, Flatten
from keras.optimizers import Adam
from keras import losses
from keras import backend as K

#Now, using Keras with tensorflow backend, we build a DCGAN object 
#(Deep Convolutional GAN), in which we define the generator and discriminator CNN and how to train them.
class DCGAN:
    def __init__(self, output_directory, img_size):
        # some parameters
        self.img_size = img_size  #dimension of the input patches in pixels
        self.channels = 1         #number of channel (in this case 1)
        self.kernel_size = 3      #we fix the filters dimension being 3x3
        self.output_directory = output_directory   #directory where to save stuff
        self.latent_dim = 100     # dimension of the noise vector input to the generator

    def smooth_accuracy(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation('tanh'))
        noise = Input(shape=(self.latent_dim,))
        img_out = model(noise)
        return Model(noise, img_out)

 def build_discriminator(self):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=1, input_shape=img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        img = Input(shape=img_shape)
        validity = model(img)
        return Model(img, validity)

 def build_gan(self):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        z = Input(shape=(self.latent_dim,))
        fake_img = self.generator(z)
        self.discriminator.trainable = False  #we want to update only G weights
        label = self.discriminator(fake_img)
        self.combined = Model(z, label)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


def train(self, epochs, patches_file, batch_size=32, save_interval=100, swap=None, seed=4324, str_run='test'):
        self.build_gan()
        Y_train = np.load(patches_file)['patches']
        print("Training Data Shape: ", Y_train.shape)
        half_batch = batch_size // 2
        accs = []
        for epoch in range(epochs):
            ind_batch = np.random.randint(0, Y_train.shape[0], batch_size)
            ind_hbatch = np.random.randint(0, Y_train.shape[0], half_batch)
            input_noise = np.random.randn(Y_train.shape[0], self.latent_dim)
            input_noise_batch =  np.random.randn(batch_size, self.latent_dim)
            input_noise_hbatch =  np.random.randn(half_batch, self.latent_dim)
            target_fake_gan = np.ones((batch_size, 1))
            
            #traing generator on N fake images
            g_loss = self.combined.train_on_batch(input_noise_batch, target_fake_gan)
            
            real_imgs = Y_train[ind_hbatch]
            real_imgs = real_imgs.reshape((half_batch, self.img_size[0], self.img_size[1], 1))
            fake_imgs = self.generator.predict(input_noise_hbatch)
            target_real_disc = np.ones((half_batch, 1))
            target_fake_disc = np.zeros((half_batch, 1))
            

 # train discriminator on N/2 real and N/2 fake images
            d_loss_real = self.discriminator.train_on_batch(real_imgs, target_real_disc)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, target_fake_disc)
            # save progress
            if epoch % (save_interval) == 0:
                print('Epoch: ', epoch)
                save_path = self.output_directory + "/models"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.discriminator.save(save_path + '/discrim_'+str_run+str(epoch)+'.h5')
                self.generator.save(save_path + '/generat_'+str_run+str(epoch)+'.h5')
                gen_imgs_tot = self.generator.predict(input_noise)
        self.discriminator.save(save_path + '/discrim_'+str_run+str(epoch)+'.h5')
        self.generator.save(save_path + '/generat_'+str_run+str(epoch)+'.h5')

dcgan = DCGAN(output_directory='./', img_size=(64, 64))


dcgan.train(epochs=2001, patches_file='cmb_patches_1000.npz', batch_size=32, save_interval=1000, str_run='cmb', swap=5)

test_noise = np.random.randn(1000, 100)
gen_imgs_test = dcgan.generator.predict(test_noise)

gener_pretrained = load_model('./generat_cmb_pretrained.h5')
discrim_pretrained = load_model('./discrim_cmb_pretrained.h5')
gen_imgs_test = gener_pretrained.predict(test_noise)

gen_imgs_test.shape
cmb_nn = gen_imgs_test[:, :, :, 0]

sel = np.random.randint(0, 1000)
cmb_sfl = [cmb_res[sel], cmb_nn[sel]]
sfl = [0, 1]
np.random.shuffle(sfl)
plt.subplot(121)
plt.imshow(cmb_sfl[sfl[0]], vmin=-1, vmax=1)
plt.subplot(122)
plt.imshow(cmb_sfl[sfl[1]], vmin=-1, vmax=1)

if sfl == [0,1]:
    print('Real, Fake')
else:
    print('Fake, Real')

cmb_res = np.load('cmb_patches_1000.npz')['patches']

cl_nn_tot = []
for i in range(1000):
    ell, cl_nn = calculate_2d_spectrum(cmb_nn[i]*window, 30, 2000, reso, 64)
    cl_nn_tot.append(cl_nn)

cl_nn_mean = np.mean(cl_nn_tot, axis=0)
cl_data_mean = np.mean(cl_data_tot, axis=0)
plt.semilogy(ell, cl_data_mean*ell*(ell+1)*bias, 'o')
plt.semilogy(ell, cl_nn_mean*ell*(ell+1)*bias, 'o')
plt.semilogy(cl*l*(l+1))
plt.xlim(2, 1000)
plt.ylim(1e3, 1e5)


#generating Matter power spectrum from the fake data 

import sys, platform, os
import numpy as np
import camb
import yaml
from camb import model, initialpower
#from astropy.io import ascii    

print("Parsing parameters...")
parFile = open(sys.argv[1], 'r')
inputPars = yaml.load(parFile)

print("Setting up cosmology...")
pars = camb.CAMBparams()
pars.set_cosmology(H0=inputPars['H_0'], ombh2=inputPars['Omega_bh2'], omch2=inputPars['Omega_ch2'],
                   omk=inputPars['Omega_k'], tau=inputPars['tau'])

print("Calculating...")
results = camb.get_results(pars)

print("Getting non-linear power spectrum...")
pars.set_dark_energy()
pars.set_matter_power(redshifts=[inputPars['z']], kmax=inputPars['k_max'])
pars.InitPower.set_params(ns=inputPars['n_s'])
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=inputPars['k_min'], maxkh=inputPars['k_max'], 
                                              npoints=inputPars['num_points'])
sigma_8 = np.array(results.get_sigma8())
print (sigma_8)

print("Outputting non-linear power spectrum...")
#ascii.write([kh, pk[0]], inputPars['outFile'], names=['kh', 'P(k)'], overwrite=True)
np.savetxt(inputPars['outFile'], [kh, pk[0]])

print("Getting non-linear power spectrum...")
pars.NonLinear = model.NonLinear_none
results.calc_power_spectra(pars)
kh_lin, z, pk_lin = results.get_matter_power_spectrum(minkh=inputPars['k_min'], maxkh=inputPars['k_max'], 
                                              npoints=inputPars['num_points'])

print("Outputting non-linear power spectrum...")
#ascii.write([kh_lin, pk_lin[0]], inputPars['outLinFile'], names=['kh', 'P(k)'], overwrite=True)
np.savetxt(inputPars['outLinFile'], [kh_lin, pk_lin[0]])

sigma_8 = np.array(results.get_sigma8())
print (sigma_8)
