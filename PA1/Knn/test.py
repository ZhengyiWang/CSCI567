import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)   
    
    tp=sum([x==1 and y==1 for x,y in zip(predicted_labels,real_labels)])
    fp=sum([x==1 and y==0 for x,y in zip(predicted_labels,real_labels)])
    fn=sum([x==0 and y==1 for x,y in zip(predicted_labels,real_labels)])
    
    if 2*tp+fp+fn==0:
        return 0
    f1=2*tp/(2*tp+fn+fp)
    return f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        
        value=0
        for x,y in zip(point1,point2):
            value=value+pow(abs(x-y),3)
        return pow(value,1/3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        
        value=0
        for x,y in zip(point1,point2):
            value=value+pow((x-y),2)        
        return pow(value,1/2)

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
		
        value=[x*y for x,y in zip(point1,point2)]
        return sum(value)		

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
		
        value=x_value=y_value=0
        for x,y in zip(point1,point2):
            x_value=x_value+pow(x,2)
            y_value=y_value+pow(y,2)
            value=value+x*y
        if pow(x_value,1/2)*pow(y_value,1/2)==0:
            return 0
        else:
            distance=1-value/(pow(x_value,1/2)*pow(y_value,1/2))
            return distance
		

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
       
        value=0
        for x,y in zip(point1,point2):
            value=value+pow((x-y),2)
        return -np.exp(-0.5*value)  


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        
        f1_score_records=[]
        for k in range(1, min(len(x_train),30),2):
            f1_score_records.append([])
            for distance_function in list(distance_funcs.values()):
                model=KNN(k,distance_function)
                model.train(x_train,y_train)
                predicted_labels=model.predict(x_val)
                f1_score_records[(k-1)//2].append(f1_score(y_val,predicted_labels))
        
        best_model_list=np.where(f1_score_records==np.max(f1_score_records))
        best_model_index=zip(best_model_list[1],best_model_list[0])
        index=sorted(best_model_index,key=lambda x:(x[0],x[1]))[0]
        
        
        self.best_distance_function=list(distance_funcs.keys())[index[0]]
        self.best_k=index[1]*2+1
        self.best_model=KNN(self.best_k,distance_funcs[self.best_distance_function])
        self.best_model.train(x_train,y_train)
        
        
    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        
        f1_score_records=[]
        for k in range(1, min(len(x_train),30), 2):
            f1_score_records.append([])
            for scaler in list(scaling_classes.values()):
                tmp=[]
                fun=scaler()
                for distance_function in list(distance_funcs.values()):
                    model=KNN(k,distance_function)
                    model.train(fun(x_train),y_train)
                    predicted_labels=model.predict(fun(x_val))
                    tmp.append(f1_score(y_val,predicted_labels))
                f1_score_records[(k-1)//2].append(tmp)
        
        best_model_list=np.where(f1_score_records==np.max(f1_score_records))
        best_model_index=zip(best_model_list[1],best_model_list[2],best_model_list[0])
        index=sorted(best_model_index,key=lambda x:(x[0],x[1],x[2]))[0]
        
        self.best_distance_function=list(distance_funcs.keys())[index[1]]
        self.best_k=index[2]*2+1
        self.best_scaler=list(scaling_classes.keys())[index[0]]
        self.best_model=KNN(self.best_k,distance_funcs[self.best_distance_function])
        self.best_model.train(x_train,y_train)


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        new_features=[]
        for vector in features:
            denominator=pow(sum([x*x for x in vector]),1/2)
            if denominator==0:
                tmp=vector
            else:
                tmp=[x/denominator for x in vector]
            new_features.append(tmp)
        
        return new_features


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_check=True
        

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        if self.first_check:
            self.maxes=np.max(features, axis=0)
            self.minis=np.min(features, axis=0)
            new_features=((features-self.minis)/(self.maxes-self.minis)).tolist()
            self.first_check=False
            return new_features
        else:
            new_features=((features-self.minis)/(self.maxes-self.minis)).tolist()
            return new_features