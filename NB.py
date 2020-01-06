import numpy as np
import pandas as pd
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('punkt')

# Preprocessing of the dataset

df=np.load('data_train.pkl', allow_pickle=True)

labels=np.unique(df[1])
# print(labels[16])
valid_size=2000
train_set=[df[0][:-valid_size],df[1][:-valid_size]]
validation_set=[df[0][-valid_size:],df[1][-valid_size:]]

    # Number of exemples
n=len(train_set[0])

    # Process the data

data=[]

    # Remove the uppercases, stopwords, the numerics and the "\'"
print('step0')
words=set(stopwords.words('english'))
bow=[]
def processing(message):
    r=[]
    word_tk=word_tokenize(message)
    for J in word_tk:
        j=J.lower()
        if j not in words and 'a'<j<'zz' and '\'' not in j:
            r.append(j)
            bow.append(j)
    return r

for i in range(n):
    data.append(processing(train_set[0][i]))
print('step1')
full_bow=np.unique(bow,return_counts=True)
print('step2')
# threshold=max(bow[1])*10//100
# final_bow= bow[0][bow[1]>threshold]
# final_count=bow[1][bow[1]>threshold]
# print(len(final_bow))

###

# Defining our features

tags=np.array([])
for lab in labels:
    data_lab=[]
    for i in range(n):
        if train_set[1][i]==lab: data_lab+=data[i]

    unique_values=2*np.unique(np.append(data_lab,full_bow[0]),return_counts=True)[1]-full_bow[1]-1
    threshold=max(unique_values)*0.4
    # print(threshold)
    lab_tag=full_bow[0][unique_values>threshold]
    tags=np.append(tags,lab_tag)
    # print(lab,lab_tag)
tags=np.unique(np.append(tags,labels))
# print('The number of features is ',len(tags))
print('step3')
# Express the messages in terms of the features

def hist(text):
    length=len(text)
    histogram=np.ones((length,len(tags)))/len(tags)
    for i in range(length):
        for j in range(len(tags)):
            if tags[j] in text[i]:
                outvote=1
                if tags[j] in labels: outvote=10
                histogram[i,j]+=outvote*sum(np.array(text[i])==tags[j])
    return histogram
histogram=hist(data)

# Defining our probabilities

tag_freq=np.zeros((len(labels),len(tags)))
for i in range(len(labels)):
    tag_freq[i]=sum(histogram[np.array(train_set[1])==labels[i]])
tag_freq+=1
tag_freq/=tag_freq.sum(axis=0)
# print(tag_freq)

# nb_classes=len(final_bow)

# Define the sparse vectors
#
# data_vec=np.zeros((n,nb_classes))
# for i in range(nb_classes):
#     for j in range(len(data)):
#         data_vec[j,i]=data[j].count(final_bow[i])
#
# print(data_vec)

#
# class GaussianMaxLikelihood:
#     def __init__(self, n_dims):
#         self.n_dims = n_dims
#         self.mu = np.zeros(n_dims)
#         self.sigma_sq = 1.0
#
#     # For a training set, the function should compute the ML estimator of the mean and the variance
#     def train(self, train_data):
#         self.mu = np.mean(train_data, axis=0)
#         self.sigma_sq = np.sum((train_data - self.mu) ** 2.0) / (self.n_dims * train_data.shape[0])
#
#     # Returns a vector of size nb. of test ex. containing the log probabilities of each test example under the model.
#     def loglikelihood(self, test_data):
#         c = - self.n_dims * np.log(2 * np.pi) / 2 - self.n_dims * np.log(np.sqrt(self.sigma_sq))
#         log_prob = c - np.sum((test_data - self.mu) ** 2.0, axis=1) / (2.0 * self.sigma_sq)
#         return log_prob
#

class BayesClassifier:
    def __init__(self, probs, priors):
        self.probs = probs
        self.priors = priors
        if (self.probs).shape[0] != len(self.priors):
            print('The number of probability models must be equal to the number of priors!')

    def predictions(self, test_vec):
        freq_test=np.zeros((len(self.priors),len(test_vec)))
        for j in range(len(test_vec)):
            for i in range(len(self.priors)):
                freq_test[i,j]=sum(np.log(test_vec[j]*self.probs[i]))+np.log(self.priors[i])
        return labels[np.argmax(freq_test,axis=0)]

print('step4')
priors=1/(np.unique(train_set[1],return_counts=True)[1])

modele=BayesClassifier(tag_freq,priors)
val_test=[]
for i in range(len(validation_set[0])):
    val_test.append(processing(validation_set[0][i]))
freq_val=hist(val_test)
print(freq_val.argmax(axis=1))
pred=modele.predictions(freq_val)
print(pred)

def precision(result, labels):
    if len(result)!=len(labels):
        print('Sizes don\'t match')
        return 0
    count=0
    for i in range(len(result)):
        if result[i]==labels[i]: count+=1
    return count/len(result)

score=precision(pred,validation_set[1])

print('The score on the validation set is : ',score*100, '%')