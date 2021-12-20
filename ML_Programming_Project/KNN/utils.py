import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
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
    real_total = 0
    predicted_total = 0
    total = 0
    for realValue, predictedValue in zip(real_labels, predicted_labels):
        total = total + realValue * predictedValue
        real_total = real_total + realValue
        predicted_total = predicted_total + predictedValue
    return 2 * (float(total) / float(real_total + predicted_total))
    raise NotImplementedError


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
        cubed_distance = 0
        for x, y in zip(point1, point2):
            cubed_distance = cubed_distance + np.absolute(x - y) ** 3
        m_distance = float(np.cbrt(cubed_distance))
        return m_distance
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        squared_distance = 0
        for x, y in zip(point1, point2):
            squared_distance = squared_distance + (x - y) ** 2
        e_distance = float(np.sqrt(squared_distance))
        return e_distance
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        p1Norm = 0
        p2Norm = 0
        total = 0
        for x, y in zip(point1, point2):
            p1Norm = p1Norm + x ** 2
            p2Norm = p2Norm + y ** 2
            total = total + x * y
        if p1Norm == 0 or p2Norm == 0:
            return 1
        else:
            c_distance = float(1 - float(total) / float(np.sqrt(p1Norm) * np.sqrt(p2Norm)))
            return c_distance
        raise NotImplementedError



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        best_f1_score = 0
        for distances in distance_funcs:
            for k in range(1, min(31, len(x_train) + 1), 2):
                model = KNN(k, distance_funcs[distances])
                model.train(x_train, y_train)
                y_val_pred = model.predict(x_val)
                calculated_f1_score = f1_score(y_val, y_val_pred)
                if calculated_f1_score > best_f1_score:
                    self.best_k = k
                    self.best_distance_function = distances
                    self.best_model = model
                    best_f1_score = calculated_f1_score

        # raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        best_f1_score = 0
        for scaled_func in scaling_classes:
            scalar = scaling_classes[scaled_func]()
            scaled_x_train = scalar.__call__(x_train)
            scaled_x_val = scalar.__call__(x_val)
            for distances in distance_funcs:
                for k in range(1, min(31, len(x_train) + 1), 2):
                    model = KNN(k, distance_funcs[distances])
                    model.train(scaled_x_train, y_train)
                    y_val_pred = model.predict(scaled_x_val)
                    calculated_f1_score = f1_score(y_val, y_val_pred)
                    if calculated_f1_score > best_f1_score:
                        self.best_k = k
                        self.best_distance_function = distances
                        self.best_scaler = scaled_func
                        self.best_model = model
                        best_f1_score = calculated_f1_score
        # raise NotImplementedError


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
        no_of_data = len(features)
        feature_num = len(features[0])
        features_norm = [[0] * feature_num for x in range(no_of_data)]
        for x in range(no_of_data):
            norm = 0
            for y in range(feature_num):
                norm += (features[x][y] ** 2)
            norm = np.sqrt(norm)
            if norm == 0:
                features_norm[x] = features[x]
                continue
            for y in range(feature_num):
                features_norm[x][y] = 0 if features[x][y] == 0 else features[x][y] / norm
        return features_norm
        raise NotImplementedError


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        no_of_data = len(features)
        feature_num = len(features[0])
        features_norm = [[0] * feature_num for _ in range(no_of_data)]
        minFeatures = [float("inf")] * feature_num
        maxFeatures = [float("-inf")] * feature_num

        for y in range(no_of_data):
            for x in range(feature_num):
                val = features[y][x]
                maxFeatures[x] = max(maxFeatures[x], val)
                minFeatures[x] = min(minFeatures[x], val)

        for y in range(no_of_data):
            for x in range(feature_num):
                MinMaxdiff = maxFeatures[x] - minFeatures[x]
                features_norm[y][x] = 0 if MinMaxdiff == 0 else ((features[y][x] - minFeatures[x]) / MinMaxdiff)
        return features_norm
        raise NotImplementedError
