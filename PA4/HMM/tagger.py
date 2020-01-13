import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
	###################################################
	# Edit here
    
    state_dict=dict()

    for i in range(len(tags)):
        if not tags[i] in state_dict.keys():
            state_dict[tags[i]]=i
        

    pi=np.zeros(len(tags))
    
    for t in train_data:
        ind=state_dict[t.tags[0]]
        pi[ind]=pi[ind]+1
           
    for x in range(len(tags)):
        pi[x]=pi[x]/len(train_data)
    
    
    obs_dict=dict()
    ind=0
    
    for t in train_data:
        sentence=t.words
        for word in sentence:
            if word not in obs_dict.keys():
                obs_dict[word]=ind
                ind=ind+1

    
    A=np.zeros([len(tags),len(tags)])
    start=np.zeros(len(tags))
    
    for t in train_data:
        sentence=t.tags
        for j in range(len(sentence)-1):
            s=sentence[j]
            start[state_dict[s]]=start[state_dict[s]]+1
            sp=sentence[j+1]
            A[state_dict[s]][state_dict[sp]]=A[state_dict[s]][state_dict[sp]]+1

    for i in range(len(tags)):
        if start[i]!=0:
            A[i]=[x/start[i] for x in A[i]]
    
    
    B=np.zeros([len(tags),ind])
    
    for t in train_data:
        start[state_dict[t.tags[-1]]]=start[state_dict[t.tags[-1]]]+1
        for i in range(len(t.tags)):
            s=t.tags[i]
            o=t.words[i]
            B[state_dict[s]][obs_dict[o]]=B[state_dict[s]][obs_dict[o]]+1
    
    for i in range(len(tags)):
        if start[i]!=0:
            B[i]=[x/start[i] for x in B[i]]
    
    model=HMM(pi,A,B,obs_dict,state_dict)
	###################################################
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging=[]
	###################################################
	# Edit here
    
    ind=max(model.obs_dict.values())+1
    z=np.full((len(tags),1),1e-6)
    
    for t in test_data:
        for word in t.words:
            if word not in model.obs_dict.keys():
                model.obs_dict[word]=ind
                model.B=np.append(model.B,z,axis=1)
                ind=ind+1
        tagging.append(model.viterbi(t.words))

	###################################################
    return tagging
