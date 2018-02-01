#!/usr/bin/env python2
"""Stochastic Gradient Descent for Logistic Regression.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from math import exp
import random


# TODO: Calculate logistic
def logistic(x):
    f = exp(x)/(1+exp(x))
    return f

# TODO: Calculate dot product of two lists
def dot(x, y):
    s = 0
    for index in range(0,len(x)):
        s = s + x[index]*y[index]
    return s

# TODO: Calculate prediction based on model
def predict(model, point):
    s = dot(model, point['features'])
    predicted_value = logistic(s)
    return predicted_value

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    correct = 0
    for index in range(0,len(data)):
        y = float(data[index]['label'])
        p = predictions[index]
        if(p>0.5):
            p = float(1)
        else:
            p = float(0)
        if(y==p):
            correct += 1
    return float(correct)/len(data)
    

# TODO: Update model using learning rate and L2 regularization
def update(model, point, delta, rate, lam): 
    grad = [x*delta for x in point]
    reg = [x*(-lam) for x in model]
    gradient = [x+y for x,y in zip(reg,grad)]
    step = [x*rate for x in gradient]
    model = [x+y for x,y in zip(model,step)]  # model = model + step*(reg + grad)

    return model

def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

# TODO: Train model using training data
def train(data, epochs, rate, lam):
    tol = 0.00001
    model = initialize_model(len(data[0]['features']))
    for runs in range(0,epochs):
        for index in range(0,len(data)):
            lst = data[index]['features']
            point = [float(i) for i in lst]
            inner_prod = dot(model,point)
            prediction = logistic(inner_prod)
            delta = float(data[index]['label']) - prediction
            model_new = update(model,point, delta, rate,lam)
            if (abs(sum(model) - sum(model_new))<= tol):
                break
            model = model_new
        random.shuffle(data)
        print('epoch=',runs)
    return model

def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(r['marital'] == 'Married-civ-spouse')
        #TODO: Add more feature extraction rules here!
        features.append(float(r['hr_per_week'])/100)
        features.append(float(r['capital_gain'])/100000)
        features.append(float(r['capital_loss'])/100000)
        
        point['features'] = features
        data.append(point)
    return data

################## Transform to one-hot vector ######################
def transform_to_hot_vector(feature_data):
        vec = feature_data[1:]
        unique_vector = sorted(set(vec))
        dimension = len(unique_vector)
        categorical_data_array = []
        for i in range(0,len(vec)):
            val = feature_data[i] 
            for k in range(0,len(unique_vector)):
                if val == unique_vector[k]:
                    categorical_data_array.append([1.0 if i == k else 0.0 for i in range(dimension)])
        return categorical_data_array
#####################################################################
# TODO: Tune your parameters for final submission
def submission(data):
    return train(data, 20, .5, 0.000015)
