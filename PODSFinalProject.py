#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:53:38 2024

@author: rithvikkottapalli
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from scipy.stats import kruskal

N_NUMBER = 10646862
np.random.seed(N_NUMBER)

df = pd.read_csv('spotify52kData.csv')
pd.set_option('display.max_columns', None)   
#print(df)

dflist = df.values
array = np.array(dflist)
#print(array)

###############################################################
#Problem 1: EDA
'''
numerical_columns = [5,6,8,9,11,13,14,15,16,17,18] #Columns that could be suspected to be normally distributed (number not index)
numBins = 51  #arbitrary but large enough
plt.hist(array[:,5-1], bins=numBins) #change the number to one in the numerical columns array to see dist of each variable
'''

###############################################################
#Problem 2: Relationship between length and popularity
'''
var1 = 6-1 #duration
var2 = 5-1 #popularity
r = df['duration'].corr(df['popularity'])
print(r)
plt.xlim(1000000)
plt.gca().invert_xaxis()
plt.plot(array[:,var1],array[:,var2],'x') #Make a scatter plot with x's as markers
plt.xlabel('Duration') #Suitable x-axis label
plt.ylabel('Popularity') #Suitable y-axis label
plt.title(f'r = {r}') #Title 
'''

###############################################################
#Problem 3: Explicit Songs more popular?
'''
df1 = df.iloc[:, 6] #explicit
df2 = df.iloc[:, 4] #popularity
df_combined = pd.concat([df1, df2], axis=1)
dflist = df_combined.values
explicit = []
non_explicit = [] #these lists will hold the popularities of their populations
for i in dflist:
    if i[0]:
        explicit.append(i[1])
    else:
        non_explicit.append(i[1])

u1,p1 = stats.mannwhitneyu(non_explicit,explicit)

print(u1)
print(p1)
print(np.median(explicit))
print(np.median(non_explicit))
print(np.mean(explicit))
print(np.mean(non_explicit))
'''

###############################################################
#Problem 4: Major more popular than minor?
'''
df1 = df.iloc[:, 11] #mode
df2 = df.iloc[:, 4] #popularity
df_combined = pd.concat([df1, df2], axis=1)
dflist = df_combined.values
major = []
minor = [] #these lists will hold the popularities of their populations
for i in dflist:
    if i[0] == 1:
        major.append(i[1])
    else:
        minor.append(i[1])

u1,p1 = stats.mannwhitneyu(major,minor)

print(u1)
print(p1)
print(np.median(major))
print(np.median(minor))
print(np.mean(major))
print(np.mean(minor))
print(len(major))
print(len(minor))
'''

###############################################################
#Problem 5: Energy Correlated with Loudness?
'''
var1 = 9-1 #energy
var2 = 11-1 #loudness
r = df['energy'].corr(df['loudness'])
print(r)
plt.plot(array[:,var1],array[:,var2],'x') #Make a scatter plot with x's as markers
plt.xlabel('Energy') #Suitable x-axis label
plt.ylabel('Loudness') #Suitable y-axis label
plt.title(f'r = {r}')

'''

###############################################################
#Problem 6: Single Best Predictor
'''
numerical_columns_index = [5,7,8,10,12,13,14,15,16,17]

for i in numerical_columns_index:
    X = array[:,i].reshape(len(array),1)
    y = array[:,4] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    singleFactorModel = LinearRegression().fit(X_train,y_train) 
    
    rSqrSingle = singleFactorModel.score(X_test, y_test) 
    print(i, rSqrSingle)
    
    slope = singleFactorModel.coef_ 
    intercept = singleFactorModel.intercept_ # And B0 (intercept)
 
    yHat = slope * array[:,i] + intercept
    plt.plot(array[:,i],array[:,4],'o',markersize=3) 
    plt.xlabel('Instrumentalness') 
    plt.ylabel('Popularity')  
    plt.plot(array[:,i],yHat,color='orange',linewidth=3)
    plt.title('Using scikit-learn: R^2 = {:.3f}'.format(rSq)) 
'''

###############################################################
#Problem 7: Prediction with all variables

'''
numerical_columns_index = [5,7,8,10,12,13,14,15,16,17]

X = array[:, numerical_columns_index]
y = array[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

allFactorModel = LinearRegression().fit(X_train,y_train) 
rSqrAll = allFactorModel.score(X_test,y_test) 
print(rSqrAll)
'''
###############################################################
#Problem 8: PCA
'''
numerical_columns_index = [5,7,8,10,12,13,14,15,16,17]
numericalArray = array[:, numerical_columns_index]


# 0 Duration
# 1 Danceability
# 2 Energy
# 3 Loudness
# 4 Speechiness
# 5 Acousticness
# 6 Instrumentalness
# 7 Liveness
# 8 Valence
# 9 Tempo


numericaldf = pd.DataFrame(numericalArray)
corrMatrix = numericaldf.corr()

# Plot the data:
plt.imshow(corrMatrix, cmap='plasma') 
plt.xlabel('Feature')
plt.ylabel('Feature')
plt.colorbar()
plt.show()

numericaldf_zscored = (numericaldf - numericaldf.mean()) / numericaldf.std()

print(numericaldf_zscored)

zscoredArray = np.array(numericaldf_zscored.values)
pca = PCA().fit(zscoredArray)

eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredArray)
varExplained = eigVals/sum(eigVals)*100

for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))
    
numFeatures = 10
x = np.linspace(1,numFeatures,numFeatures)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numFeatures],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

whichPrincipalComponent = 2 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Feature')
plt.ylabel('Loading')
plt.show() # Show bar plot

#Factor 1: 3,4, -6   
#Factor 2: 2, 9     minus implies negative correlation, features not 0 indexed
#Factor 3: -5, -8

print(rotatedData)
'''

###############################################################
#Problem 9: Predict Major/Minor from Valence
'''
numerical_columns_index = [5,7,8,10,12,13,14,15,16,17]

current = 16
plt.scatter(array[:,current],array[:,11],color='black')
plt.xlabel('Feature')
plt.ylabel('Mode')
plt.yticks(np.array([0,1]))
plt.show()

#train test split
X = array[:,current].reshape(len(array),1) 
y = df['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit model:
model = LogisticRegression().fit(X,y)

y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities of the positive class

# Calculate AUC-ROC score
auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC-ROC Score: {auc:.4f}")


x1 = np.linspace(0,1)
y1 = x1 * model.coef_ + model.intercept_
sigmoid = expit(y1)

# Plot:
plt.plot(x1,sigmoid.ravel(),color='red',linewidth=3) # the ravel function returns a flattened array
plt.scatter(array[:,16],array[:,11],color='black')
plt.hlines(0.5,0,1,colors='gray',linestyles='dotted')
plt.xlabel('Valence')
plt.ylabel('Major?')
plt.yticks(np.array([0,1]))
plt.show()
'''

###############################################################
#Problem 10: Predicting if Classical
'''
def map_to_binary(value):
    return 1 if value == 'classical' else 0

# Apply the function to create a new column 'binary_genre'
df['track_genre'] = df['track_genre'].apply(lambda x: map_to_binary(x))
print(df)

plt.scatter(array[:,5],df['track_genre'],color='black')
plt.xlabel('Duration')
plt.ylabel('Classical?')
plt.yticks(np.array([0,1]))
plt.show()

numerical_columns_index = [5,7,8,10,12,13,14,15,16,17]
numericalArray = array[:, numerical_columns_index]
numericaldf = pd.DataFrame(numericalArray)
numericaldf_zscored = (numericaldf - numericaldf.mean()) / numericaldf.std()


zscoredArray = np.array(numericaldf_zscored.values)

#train test split
pca = PCA(n_components=3)
X = pca.fit_transform(zscoredArray)
y = df['track_genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit logistic regression on the principal components
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities of the positive class

# Calculate AUC-ROC score
auc = roc_auc_score(y_test, y_pred_proba)

print(f"AUC-ROC Score: {auc:.4f}")


'''
###############################################################
#Problem EC: Keys related to danceability?
'''
category_counts = df['key'].value_counts()
print(category_counts)
category_means = df.groupby('key')['danceability'].median()
print(category_means)
array0 = []
array1 = []
array2 = []
array3 = []
array4 = []
array5 = []
array6 = []
array7 = []
array8 = []
array9 = []
array10 = []
array11 = []

for i in array:
    if i[9] == 0:
        array0.append(i[7])
    if i[9] == 1:
        array1.append(i[7])
    if i[9] == 2:
        array2.append(i[7])
    if i[9] == 3:
        array3.append(i[7])
    if i[9] == 4:
        array4.append(i[7])
    if i[9] == 5:
        array5.append(i[7])
    if i[9] == 6:
        array6.append(i[7])
    if i[9] == 7:
        array7.append(i[7])
    if i[9] == 8:
        array8.append(i[7])
    if i[9] == 9:
        array9.append(i[7])
    if i[9] == 10:
        array10.append(i[7])
    if i[9] == 11:
        array11.append(i[7])

h,pK = stats.kruskal(array0, array1, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11)

print(h)
print(pK)
'''