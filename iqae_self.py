import numpy as np

class IQAE(object):
    def __init__(self):
        self.all_iteration_results = {}
        pass

    def estimate_probability(self, epsilon: float, alpha: float, Nshots: int, conf_interval: str = "Chernoff-Hoeffding") -> tuple[float, float]:
        """give a confidence interval for the estimate

        Args:
            epsilon (float): desired error
            alpha (float): confidence level
            Nshots (int): number of shots per iteration
            conf_interval (str, optional): type of confidence interval to use. Choose between "Clopper-Pearson" or "Chernoff-Hoeffding". Defaults to "Chernoff-Hoeffding".

        Returns:
            tuple[float, float]: [lower bound, upper bound]
        """
	
        i = 0 # initialize iteration count
        ki = 0 # initialize power of Q
        ki_previous = 0
        upi = True # keeps track of half-plane
        [theta_lower, theta_upper] = [0, np.pi/2]
        T = np.ceil(np.log2(np.pi/(8 * epsilon))) # maximum number of rounds
        
        Lmax = self.chernoff_Hoeffding_Lmax(Nshots, epsilon, alpha)
        # or 
        # Lamx = clopper_Pearson_Lmax(Nshots, epsilon, alpha)
        
        while theta_upper - theta_lower > 2 * epsilon:
            i = i+1
            ki, upi = self.findNextK(ki_previous,theta_lower, theta_upper, upi)
            Ki = 4*ki + 2
            if Ki > np.ceil(Lmax/epsilon):
                N = np.ceil(Nshots*Lmax/(epsilon * Ki * 10)) # no overshoot condition
            else:
                N = Nshots

            ai = qc_experiment(ki, N) # run circuit with power ki
            if ki == ki_previous:
                # combine results of all iterations j <=i with kj=ki into a single result
                pass

            if conf_interval == "Chernoff-Hoeffding":
                e_ai = np.sqrt(1/(2*N) * np.log(2 * T/alpha))
                a_max = min(1, ai+ e_ai)
                a_min = max(0, ai-e_ai)

            elif conf_interval == "Clopper-Pearson":
                a_max = inv_reg_beta_fn(alpha/(2*T), N * ai, N*(1-ai)+1)
                a_min = inv_reg_beta_fn(1-alpha/(2*T), N*ai+1, N*(1-ai))
        
            [theta_min, theta_max] = compute_conf_interval(a_max, a_min, upi)
            theta_lower = (np.floor((Ki * theta_lower) % (2 * pi)) + theta_min)/Ki
            theta_upper = (np.floor((Ki * theta_upper) % (2 * pi)) + theta_max)/Ki

        [a_lower, a_upper] = [np.sin(theta_lower)**2, np.sin(theta_upper)**2]
        return [a_lower, a_upper]

    def findNextK(self, k_i, theta_lower, theta_upper, up_i, r=2):
        K_i = 4*k_i + 2 # current theta factor
        theta_min = K_i * theta_lower # lower bound for scaled theta
        theta_max = K_i * theta_upper # upper bound for scaled theta
        K_max = np.floor(np.pi/(theta_upper - theta_lower)) # upper bound for theta factor
        K = K_max - (K_max - 2) % 4 # largest potential candidate of the form 4k+2
        while K >= r * K_i:
            q = K/K_i # factor to scale the interval
            if (q * theta_max) % (2*np.pi) <= np.pi and (q * theta_min) % (2*np.pi) <= np.pi:
                # interval is in the upper half-plane
                K_next = K
                up_next = True
                k_next = (K_next -2)/4
                return (k_next, up_next)
            elif (q * theta_max)% (2*np.pi) >= np.pi and (q * theta_min) % (2*np.pi) >= np.pi:
                # interval is in the lower half-plane
                K_next = K
                up_next = False
                k_next = (K_next-2)/4
                return (k_next, up_next)
            K = K -4
        return (k_i, up_i) # return old values if no new value is found

    def chernoff_Hoeffding_Lmax(self, Nshots, alpha, epsilon):
        T = np.ceil(np.log2(np.pi/(8*epsilon)))
        return (np.arcsin(2/Nshots * np.log(2*T/alpha)))**(1/4)