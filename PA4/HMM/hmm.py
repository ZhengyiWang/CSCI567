from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        
        for s in range(S):
            alpha[s][0]=self.pi[s]*self.B[s][self.obs_dict[Osequence[0]]]
            
        for t in range(1,L):
            for s in range(S):
                sum_=0
                for sp in range(S):
                    sum_=sum_+self.A[sp][s]*alpha[sp][t-1]
                alpha[s][t]=self.B[s][self.obs_dict[Osequence[t]]]*sum_
        

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for s in range(S):
            beta[s][L-1]=1
        
            
        for t in range(L-2,-1,-1):
            product_=beta[:,t+1]*self.B[:,self.obs_dict[Osequence[t+1]]]
            beta[:,t]=np.matmul(self.A,product_)
        
        
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        alpha=self.forward(Osequence)
        S=len(self.pi)
        T=len(Osequence)
        for s in range(S):
            prob=prob+alpha[s][T-1]
        
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        alpha=self.forward(Osequence)
        beta=self.backward(Osequence)
        sprob=self.sequence_prob(Osequence)
        if sprob==0:
            return prob

        prob=alpha*beta/sprob
        
        return prob
    
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha=self.forward(Osequence)
        beta=self.backward(Osequence)
        sprob=self.sequence_prob(Osequence)
        for i in range(S):
            for j in range(S):
                for t in range(L-1):
                    prob[i][j][t]=alpha[i][t]*self.A[i][j]*self.B[j][self.obs_dict[Osequence[t+1]]]*beta[j][t+1]/sprob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        path=[0]*L
        delta=np.zeros([S,L])
        big_delta=np.zeros([S,L])

        for s in range(S):
            delta[s][0]=self.pi[s]*self.B[s][self.obs_dict[Osequence[0]]]
        
        for t in range(1,L):
            for s in range(S):
                max_=0
                argmax=0
                for sp in range(S):
                    ad=self.A[sp][s]*delta[sp][t-1]
                    if max_<ad:
                        max_=ad
                        argmax=sp
                delta[s][t]=self.B[s][self.obs_dict[Osequence[t]]]*max_
                big_delta[s][t]=argmax
        
        max_=0
        argmax=0
        for s in range(S):
            if max_<delta[s][L-1]:
                max_=delta[s][L-1]
                argmax=s
        path[L-1]=argmax
        
        for tr in range(0,L-1):
            t=L-2-tr
            path[t]=int(big_delta[int(path[t+1])][t+1])
        
        key_list=list(self.state_dict.keys())
        val_list=list(self.state_dict.values())
        for t in range(L):
            val=path[t]
            path[t]=key_list[val_list.index(val)]
        ###################################################
        
        return path