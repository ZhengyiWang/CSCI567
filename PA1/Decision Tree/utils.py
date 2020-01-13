import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    
    average_entropy=0    
    total_node= sum([sum(i) for i in branches])
    
    for child in branches:
        child_entropy=0
        child_node=sum(child)
        for i in child:
            if child_node==0:
                child_entropy=0
            else:
                possibility=i/child_node
                if possibility!= 0:
                    child_entropy+=-possibility*np.log2(possibility)
        average_entropy=average_entropy+(child_node/total_node)*child_entropy
    return S-average_entropy

# TODO: implement reduced error prunning function, pruning your tree on this function
    
def prunning_recursively(decisionTree,node,X_test,y_test):
    if node.splittable==True:
        for children in node.children:
            prunning_recursively(decisionTree,children,X_test,y_test)
            
        prediction_before=decisionTree.predict(X_test)
        error_before=0
        for i in range(len(y_test)):
            if prediction_before[i]!=y_test[i]:
                error_before=error_before+1
        
        #calculate the prunning_tree        
        node.splittable=False
        prediction_after=decisionTree.predict(X_test)
        
        error_after=0
        for i in range(len(y_test)):
            if prediction_after[i]!=y_test[i]:
                error_after=error_after+1
                
        if error_before<error_after:
            node.splittable=True

def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    
    prunning_recursively(decisionTree,decisionTree.root_node,X_test,y_test)


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: '+str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')