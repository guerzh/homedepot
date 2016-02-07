from pandas import *
import os
import math
from numpy import *

os.chdir("/home/guerzhoy/Desktop/homedepot")

descs = read_csv("product_descriptions.csv")
train = read_csv("train.csv")
test = read_csv("test.csv")
attr = read_csv("attributes.csv")

train = train.merge(descs, on=["product_uid"]).merge(attr, on="product_uid")
test = test.merge(descs, on=["product_uid"]).merge(attr, on="product_uid")




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
    
skip = 50

train_learn_x = zeros((len(train.index)/skip,10))
train_learn_y = zeros((len(train.index)/skip,1))

brands = set([a.split()[0].lower() for a in train["product_title"]])


for i in range(0, len(train.index), skip):
    if i % 10000 == 0:
        print i
    line = train.loc[i,:]
    
    try:
        search_terms = [a.strip() for a in line["search_term"].lower().split()]
        product_title = [a.strip() for a in line["product_title"].lower().split()]
        product_desc = [a.strip() for a in line["product_description"].lower().split()]
        name = [a.strip() for a in line["name"].lower().split()]
        attr = [a.strip() for a in line["value"].lower().split()]
        
        relevance = float(line["relevance"])
        
        brand_present = get_brand_present(search_terms, product_title)
        other_brand_present = get_other_brand_present(search_terms, product_title, brands)
        
        train_learn_x[i/skip,:] = array((get_cosine_similarity(search_terms, product_title),
                                get_cosine_similarity(search_terms, product_desc),
                                get_cosine_similarity(search_terms, name),
                                get_cosine_similarity(search_terms, attr),
                                get_overlap(search_terms, product_title),
                                get_overlap(search_terms, product_desc),
                                get_overlap(search_terms, name),
                                get_overlap(search_terms, attr),
                                brand_present,
                                other_brand_present))
    
        train_learn_y[i/skip,:] = relevance
    except:
        print "ERROR", i
        #train_learn_x[i,:] = zeros((1, 4))
        #train_learn_y[i,0] = 0
        





from sklearn import datasets, linear_model

regr = linear_model.Ridge(alpha=.0001)
regr.fit(train_learn_x, train_learn_y)
print 'Coefficients: \n', regr.coef_
#the coefficients for the eight features computed in lines 52-59

print "RMSE: %f" % sqrt(mean((regr.predict(train_learn_x) - train_learn_y) ** 2))  #.505
print "Baseline: %f" % sqrt(mean((mean(train_learn_y) - train_learn_y) ** 2))      #.521
