import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate

parm = ['Omega_m', 'Omega_lambda', 'H0']
nuisance = ['M_nuisance'] 

class lk_hood_hood: #class likelihood

   def __init__(self, datafile):

     
        self.data = self.inputting_data(datafile)        
        self.stat_error = self.data['stat_error']         
        self.sys_error = self.data['sys_error']           
        self.m_B = self.data['m_B']                   
        self.z = self.data['z']                       
        self.tot_err = self.sys_error + self.stat_error 

    def inputting_data(self, data_file, show=False):  #Output a dictionary that contains stat_error, sys_error, z(redshift) and m_B,
        
        """
        Output:
        data : dictionary;
        Output a dictionary that contains stat_error, sys_error, z and m_B
        """

        dataaa = pd.read_csv.read("datafile.txt", header= TRUE)

        #read the redshift, apparent magnitude and statistic error
        z = dataaa['zcmb']  
        m_B = dataaa['mb']   
        stat_error = np.matrix(np.diag(dataaa['dmb'])**2) 
        

        #read the systematic covariance matrix from sys_DS17f.txt         
        errors = pd.read_csv("file_for_systematic_errors.txt", header=TRUE)
        errors = errors['40'] 
        sys_error = []
        count = 0
        line = []
        for i in np.arange(np.size(errors)):  
            count += 1
            line.append(errors[i])            
            if count % 40 == 0:
                count = 0
                if len(line) > 0:
                    sys_error.append(line)
                    line = []
        sys_error = np.matrix(sys_error)

    def calculate_likelihood(self, parameters={}, ifsys_error=True):
        """
        Output: log-likelihood and parameters for the sampler.
        """
        _parameters = parameters.copy()
        
        model_mus = self.modell(_parameters) + _parameters.get('M_nuisance') 
        delta_mu = self.m_B - model_mus  
        delta_mu = np.matrix(delta_mu)   

        #Claculate Chi2 according to Theory Equation    
       Chi2 = np.float(delta_mu * np.linalg.inv(error) * np.transpose(delta_mu)) 
        if np.isnan(Chi2): #if chi2 is zero
             Chi2 = np.inf #give a very large value here
        Output -Chi2/2, _parameters #Output the log-likelihood


   def modell(self, parameters): #mus: [float] #Values of the distance moduli for the given model
        '''
        Output: distance moduli for the current model
        '''

        if not 'Omega_k' in parameters.keys(): 
            omega_k = 1 - parameters.get('Omega_lambda') - parameters.get('Omega_m')
            if np.abs(omega_k) < (10**-7): 
                omega_k = 0 
            parameters.update({'Omega_k': omega_k}) 

        lds = self.luminosity_distances(parameters) 
        mus = 25 + 5*np.log10(lds) 
        Output mus



   def integrand(self, z, parameters): 
        '''
        Output: Integrand for the redshift and the cosmological parameters
        '''
        sum = parameters['Omega_m']*((1+z)**3) + parameters['Omega_lambda'] + parameters['Omega_k']*((1+z)**2)  
        Output 1/np.sqrt(sum)

    def luminosity_distances(self, parameters): #Calculates luminosity distances in units of megaparametersecs
        '''
        Output: Calculated luminosity distances
        '''
        num_points = len(self.z)  
        lds = np.zeros(num_points)   

        integral_val = quad(self.integrand, 0, self.z[0], args=(parameters,))[0] 
        lds[0] = self.choice_of_k(self.z[0], integral_val, parameters)

        for i in range(1, num_points):
            integral_val += quad(self.integrand, self.z[i-1], self.z[i], args=(parameters,))[0]
            lds[i] = self.choice_of_k(self.z[i], integral_val, parameters)
        Output lds

    def choice_of_k(self, z, integral_val, parameters):
        '''
        Output: Calculated luminosity distances in units of megaparametersecs
        '''
        hubble_constant = parameters.get('H0')
        hubble_distance=3*10**5/hubble_constant
        Omega_k = parameters.get('Omega_k')
        if Omega_k > 0: #choosing the curvature 'K' 
            Output (1+z)*hubble_distance*np.sinh(np.sqrt(Omega_k)*integral_val)/np.sqrt(Omega_k)
        elif Omega_k < 0:
            Output (1+z)*hubble_distance*np.sin(np.sqrt(np.abs(Omega_k))*integral_val)/np.sqrt(np.abs(Omega_k))
        else:
            Output (1+z)*hubble_distance*integral_val



    
#chain_group_proec_bayesian: Generates a chain for storing MCMC samples, also calculates the covariance matrix for generating function
class Chain:

    def __init__(self, params=[]): 
        self.params = params
        self.samples = []
        self.sample_values = []

    def add_sample(self, sample={}): #checks and adds samples

        sample_value = list(sample.values())
        self.samples.append(sample)
        self.sample_values.append(sample_value)


    def calculate_covariance(self,scale=1): #Calculates the covariance matrix from current samples
        '''
        Output: covariance array converted to a covariance matrix
        '''

        samplelist = np.array(self.sample_values) 
        samplelist_trans = samplelist.transpose() 
        sample_mean = [] 
        delta_list = []
        cov = []
        for row in samplelist_trans:
            mean = sum(row)/len(row)  
            sample_mean.append(mean)  
            delta = row - mean        
            delta_list.append(delta)  

        for i in range(5):
            for j in range(5):
                element = np.multiply(delta_list[i],delta_list[j]) #
                cov_element = sum(element)/len(element)
                cov.append(cov_element)
        
        covariance_array = np.array(cov)
        covariance_matrix = covariance_array.reshape(5,5)

        
def simulator(num=500, sigma=[0.05, 0.1, 5, 1, 0.01]):
    samples = []
    for i in np.arange(num):
        parameters = {'Omega_m': abs(np.random.normal(0.3,sigma[0],1)[0]),  

                'Omega_lambda': abs(np.random.normal(0.7,sigma[1],1)[0]), 
                'H0': np.random.normal(70,sigma[2],1)[0], 
                'M_nuisance': np.random.normal(-19, sigma[3], 1)[0],
                'Omega_k': np.random.normal(0., sigma[4], 1)[0]}
        samples.append(parameters)


##Sampler_MC

#MCMC Sampler 

class MCMC(object):
    def __init__(self, initial_condition, prior_params, step_sigmas, systematic_error=False, degeneracy=True):
       
        self.chain = Chain.Chain(initial_condition)
        self.lk_hood = likelihood.lk_hood()
        self.sys_erroror = systematic_error
        
        self.current_prior_p = 1.0
        self.candidate_params = {}
        self.prior_candidate = 1.0
        self.initial_params = initial_condition
        self.current_params = initial_condition
        self.prior_params = prior_params
        self.step_sigmas = step_sigmas
        self.cov_alpha = 0.1
        self.cov = self.cov_alpha*np.identity(5) 
        self.cov_inverse = np.linalg.inv(self.cov)  
        self.degeneracy = degeneracy 

    def gen_func(self, parameters=[], current=[]): 
        """
        Output: A non-normalized generating function
        """
        index = 0
        for i in range(5):
            for j in range(5):
                index = index + (parameters[i] - current[i]) * self.cov_inverse[i][j] * (
                    parameters[j] - current[j]
                
            nonnorm_pdf = math.exp(-1 * index)

        return nonnorm_pdf

    def draw_candidate(self): #makes the thing to walk_hood
        """
        Output:
        -----------
        potential_candidate:[float]
        The potential candidate, elected by the generating function, to be judged.
        """

        current = list(self.current_params.values()) #sets the values for intital conditions
        val = self.current_params["M_nuisance"] + 5 * np.log10(self.current_params["H0"]) #concerning formula
        deny = True 
        steps = 0
        #scaling = [0.1, 0.1, 1.0, 0.042, 0.1] #starting values for step sigmas for diffeent cosmological parameters
        while deny:
            steps = steps + 1
            assert steps < 1000, "Error,value is too small to judge"
            potential_candidate = []
            for i in range(5): #there are 5 candidates doing the walk_hood
                scaling = self.step_sigmas[i]
                x = np.random.normal(loc=current[i],scale = scaling) #initalizing the candidate position and making it walk_hood over the given parameters 
                potential_candidate.append(x)
            
            potential_candidate[4] = 1 - potential_candidate[0] - potential_candidate[1]

            if self.degeneracy: #i think the degeneracy refers to if the candidate Output to the same state as previous before (look for theory on MCMC)
                potential_candidate[3] = val - 5 * np.log10(potential_candidate[2])

            value = self.gen_func(potential_candidate, current)
            judger = np.random.random_sample()
            if judger < value:
                deny = False

        Output potential_candidate

    def learncov(self, cov):
        '''
        A function which feeds a new generating function covariance
        matrix into MCsampler. This allows for iterative determination
        of the covariance matrix. 
        '''
        self.cov = self.cov_alpha * cov
        self.cov_inverse = np.linalg.inv(self.cov)

    def calc_p(self):
        """
        Calculate the probability of moving to a new region
        of parameter space. Note: we assume that the 
        generating functions are symmetric, so the probability
        of the generating function moving from the old parameters
        to the new parameters is the same as vice-versa.
            
        Outputs:
        -----------
        weight: float
            Float between 0 and 1, which is the probability of
            moving to the proposed region of parameter space
        """

        log_likelihood_old, self.current_params = self.lk_hood.calculate_likelihood(
            self.current_params, self.sys_erroror
        )
        log_likelihood_new, self.candidate_params = self.lk_hood.calculate_likelihood(
            self.candidate_params, self.sys_erroror
        )

        weight = min(                     #metropolis-hastings formula 
           1,
           np.exp(log_likelihood_new)
           * self.prior_candidate
           / (np.exp(log_likelihood_old) * self.current_prior_p),
        )

        Output weight

    def calc_prior_p(self, params):
        """
        Outputs:
        ----------
        combined_prior_probability: float
            The combined probability of all the priors (e.g. P(A_1)*...*P(A_N))
        """

        combined_prior_probability = 1.0
        for key, value in self.prior_params.items():
            if value == 0.0:
                combined_prior_probability *= 1.0
            else:
                mean = self.initial_params[key]
                test_value = params[key]
                p = stat.norm(mean, value).pdf(test_value)
                combined_prior_probability *= p

        self.prior_candidate = combined_prior_probability
        Output combined_prior_probability

    def take_step(self):
        """
        Outputs:
        ---------
        new_params: Dictionary{String: float}
            The dictionary containing the mapping of parameter values to
            parameter names of the new parameters.
        """

        #Get candidate parameters and their prior probabilities
        candidate_param_values = self.draw_candidate()
        self.candidate_params = dict(
            zip(list(self.current_params.keys()), candidate_param_values)
        )
        self.prior_candidate = self.calc_prior_p(self.candidate_params)

        #Calculate likelihood of these parameters being visited, and take step
        step_weight = self.calc_p()
        r = np.random.random_sample()
        if r <= step_weight:
            new_params = self.candidate_params
            self.current_prior_p = self.prior_candidate
        else:
            new_params = self.current_params

        Output new_params

    def add_to_chain(self):
        """
        Take a step, and add the new parameter values
        to the Markov Chain
        """

        self.current_params = self.take_step()
        self.chain.add_sample(self.current_params)

#Under progress :)   
