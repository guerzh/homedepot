from pandas import *
import os
import math
from numpy import *

os.chdir("/home/guerzhoy/Desktop/homedepot")

descs = read_csv("product_descriptions.csv")
train = read_csv("train.csv")
test = read_csv("test.csv")


train = train.merge(descs, on=["product_uid"])#.merge(attr, on="product_uid")
test = test.merge(descs, on=["product_uid"])#.merge(attr, on="product_uid")




def get_overlap(doc1, doc2):
    overlap = len(set(doc1) & set(doc2))
    return overlap
    #return len(doc1)/5
    
    
def get_cosine_similarity(doc1, doc2):
    overlap = get_overlap(doc1, doc2)
    return float(overlap)/(len(doc1)*len(doc2))
    
    
def get_brand_present(seach_terms, product_title):
    if product_title[0] in search_terms:
        return 1
    else:
        return 0
        
def get_other_brand_present(search_terms, product_title, brands):
    
    
    brand = product_title[0]
    
    
    if len((set(search_terms)-set([brand])) & brands) > 0:
        return 1
    else:
        return 0
    
skip = 1

train_learn_x = zeros((len(train.index)/skip,6))
train_learn_y = zeros((len(train.index)/skip,1))

brands = set([a.split()[0].lower() for a in train["product_title"]])


for i in range(0, len(train.index), skip):
    if i % 1000 == 0:
        print i
    line = train.loc[i,:]
    
    try:
        search_terms = [a.strip() for a in line["search_term"].lower().split()]
        product_title = [a.strip() for a in line["product_title"].lower().split()]
        product_desc = [a.strip() for a in line["product_description"].lower().split()]
        
        
        
        relevance = float(line["relevance"])
        
        brand_present = get_brand_present(search_terms, product_title)
        other_brand_present = get_other_brand_present(search_terms, product_title, brands)
        
        train_learn_x[i/skip,:] = array((get_cosine_similarity(search_terms, product_title),
                                get_cosine_similarity(search_terms, product_desc),
                                
                                
                                get_overlap(search_terms, product_title),
                                get_overlap(search_terms, product_desc),
                                
                                
                                brand_present,
                                other_brand_present))
    
        train_learn_y[i/skip,:] = relevance
    except:
        print "ERROR", i
        #train_learn_x[i,:] = zeros((1, 4))
        #train_learn_y[i,0] = 0
        





from sklearn import datasets, linear_model

#Ridge regression:
#
# minimize (SUM_i (y_i - (a0+a1*xi1+a2*xi2+...ak*xik))^2)  + alpha*|a|^2 )
#


regr = linear_model.Ridge(alpha=.00001)
regr.fit(train_learn_x, train_learn_y)
print 'Coefficients: \n', regr.coef_
#the coefficients for the eight features computed in lines 52-59

print "RMSE: %f" % sqrt(mean((regr.predict(train_learn_x) - train_learn_y) ** 2)) 
print "Baseline: %f" % sqrt(mean((mean(train_learn_y) - train_learn_y) ** 2))      

from sklearn import cross_validation
import time

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_learn_x, train_learn_y, test_size=0.4, random_state=int(time.time()))
regr = linear_model.Ridge(alpha=.00001)
regr.fit(X_train, y_train)
print "Train RMSE: %f" % sqrt(mean((regr.predict(X_train) - y_train) ** 2))  
print "Train Baseline: %f" % sqrt(mean((mean(y_train) - X_train) ** 2))      
print "Test RMSE: %f" % sqrt(mean((regr.predict(X_test) - y_test) ** 2))  
print "Test Baseline: %f" % sqrt(mean((mean(y_test) - X_test) ** 2))      





