import numpy as np
import utils as Util

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

        self.used_attr=[]
    #TODO: try to split current node
    def split(self):
        
        if self.splittable==True:
            
            max_inf_gain=-1
            smallest_attr_size=-1
            
            #calulate parent entropy
            parent_entropy=0
            for label in np.unique(self.labels):
                percent=self.labels.count(label)/len(self.labels)
                if percent>0:
                    parent_entropy=parent_entropy+(-percent*np.log2(percent))
            
            all_attr=list(range(len(self.features[0])))
            unused_attr=list(set(all_attr)-set(self.used_attr))
            
            for attr in unused_attr:
                attr_values=np.unique(np.array(self.features)[:,attr]).tolist()
                branches=np.zeros((len(attr_values),len(set(self.labels)))).tolist()
                for i in range(len(self.features)):
                    for j in range(len(attr_values)):
                        for k in range(len(set(self.labels))):
                            if(self.features[i][attr]==attr_values[j] and 
                               self.labels[i]==list(set(self.labels))[k]):
                                branches[j][k]=branches[j][k]+1
            
                info_gain=Util.Information_Gain(parent_entropy,branches)
            
                if(info_gain>max_inf_gain):
                    max_inf_gain=info_gain
                    self.dim_split=attr
                    self.feature_uniq_split=attr_values
                    smallest_attr_size=len(attr_values)
                elif(info_gain==max_inf_gain and len(attr_values)>smallest_attr_size):
                    max_inf_gain=info_gain
                    self.dim_split=attr
                    self.feature_uniq_split=attr_values
                    smallest_attr_size=len(attr_values)

            #create child nodes
            for value in self.feature_uniq_split: #attr_values
                child_features=[]
                child_labels=[]
                for index in range(len(self.features)):
                    if self.features[index][self.dim_split]==value:
                        child_features.append(self.features[index])
                        child_labels.append(self.labels[index])
                child_num_cls=len(set(child_labels))
                child=TreeNode(child_features,child_labels,child_num_cls)
                child.used_attr.extend(self.used_attr)
                child.used_attr.append(self.dim_split)
                if len(child.used_attr)==len(self.features[0]):
                    child.splittable=False
                self.children.append(child)

            # recursively call split to create rest of tree
            for child in self.children:
                if child.splittable:
                    child.split()         
        return
        
        
    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable and feature[self.dim_split] in self.feature_uniq_split:
            childi = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[childi].predict(feature)
        else:
            return self.cls_max