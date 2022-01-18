# this files is the position predictor for volleybal: https://www.kaggle.com/johnpendenque/women-volleyball-players/code
# data-visualization and preprocessing:
#
################################################################################
# importing the function libraries
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
################################################################################
# function definition
# define a function called 'outliers' which returns a list of index of outliers
# IQR = Q3-Q1
# boundary: +- 1.5*IQR
def outliers(df, ft, boundary):
    # input:
    # df: the data frame where we are looking for outliers
    # ft: the name of the feature that we are looking for outliers (string)
    # inner_fence: a boolean input if True then the boundary is +- 1.5*IQR, if false then the boundary is +- 3*IQR, the default is set to be True
    # output: a list of index of all the outliers.

    # start with calculating the quantiles
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)

    # calcualte the Interquartile range
    IQR = Q3 - Q1

    # define the upper and lower boundary for outliers.
    upper_bound = Q3 + boundary * IQR
    lower_bound = Q1 - boundary * IQR

    # collect the outliers
    out_list = df.index[(df[ft]<lower_bound) | (df[ft]>upper_bound)]

    return out_list


# define a function to find outliers for a list of futures
def outlier_ft_list(df, ft_list, inner_fence=True):
    # input:
    # df: the data frame where we are looking for outliers
    # ft: the name of the feature that we are looking for outliers (string)
    # inner_fence: if it is set to be true then the boundary is +-1.5*IQR, otherwise +-3*IQR
    # output: a list of index of all outliers for any feature in the ft_list.

    # decide whether use inner fence as outlier boundary or the outer fence.
    if inner_fence==True:
        boundary = 1.5
    else:
        boundary = 3

    out_list = []
    # find the outliers for each feature in a for loop
    for ft in ft_list:
        out_list.extend(outliers(df, ft, boundary))

    # remove the duplications
    out_list = list(dict.fromkeys(out_list))
    return out_list


################################################################################
# data visualization and pre-processing
# import the data
df = pd.read_csv(r'C:\Users\sijin wang\Documents\GitHub\Kaggle-exercise-volleyball-position-prediction\clean_data.csv')
# rough imformation of DataFrame
df.info
# see what are in the columns
print(df.columns.to_list())
# use height, weight, spike, blok as the feature to predict the variable position number
df_dropped = df.drop(['name', 'date_of_birth', 'country'], axis=1)
# check if they are Dropped
print(df_dropped.columns.to_list())
# catagorize the position Number
df['position_number'] = pd.Categorical(df['position_number'])
# plot the correlation bewteen each variable to the y
sn.pairplot(df_dropped)
# it seems that there are repetition data 3 times
# check for duplications
df.nunique()
# note taht there are 432 rows but only 143 people: there are duplications
dfunique = df.drop_duplicates(subset=['name'])
dfunique.nunique()
# pariplot again
sn.pairplot(dfunique)
# now remove all the outliners: from the pair plot we can notice that there is significant outliers for both block and spike data: there is a zero point for both
# get rid of that point
# start with the box plot to visualize the outliners
featurelist = ['height', 'weight', 'spike', 'block']
# do a box plot for each feature before removing outliers
for feature in featurelist:
    plt.figure()
    plt.title('box plot for ' + str(feature) + ' before removing outliers')
    plt.grid(False)
    dfunique.boxplot(column=[feature])
    plt.show()
# notice there are significant outliers for both spike and block
# use the function we defined to find a list of index that are outliers.
outliers = outlier_ft_list(dfunique, featurelist)
print(outliers)
# remove the outliers.
dfunique_clean = dfunique.drop(outliers)
np.shape(dfunique_clean)
# do box  plot again to visualize the data.
for feature in featurelist:
    plt.figure()
    plt.title('box plot for ' + str(feature) + ' after removing outliers')
    plt.grid(False)
    dfunique_clean.boxplot(column=[feature])
    plt.show()

# data splitting
# define the X and y: use dummy variables for y
X = dfunique_clean[['height', 'weight', 'spike', 'block']]
# y = pd.get_dummies(dfunique_clean['position_number'], columns=['position_number'])
y = dfunique_clean['position_number']
# split the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y)
# scale the features.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

###############################################################################
# model training and evaluation
# train the ML model: try knn with GridSearchCV varying the number of nearest neighbour
mknn = KNeighborsClassifier()
param_knn = {'n_neighbors':range(1, 10)}
grid_knn = GridSearchCV(mknn, param_knn)
# train the grid search with mknn.
grid_knn.fit(X_train_scaled, y_train)

# model evaluation for grid_knn
# confusion matrix
y_pred_knn = grid_knn.predict(X_test_scaled)
# confusion_matrix(y_test, y_pred_knn)
# macro f1 score.
knn_macro = f1_score(y_test, y_pred_knn, average='macro')
knn_macro
# micro f1 score
knn_micro = f1_score(y_test, y_pred_knn, average='micro')
knn_micro

# try using Kernalized Support Vector Machines
msvc = SVC()
param_svc = {'C': np.linspace(0.1, 10), 'kernel': ('linear', 'poly', 'rbf')}
grid_svc = GridSearchCV(msvc, param_svc)
# train the model using training Dataset
grid_svc.fit(X_train_scaled, y_train)
# model evaluation for SVC
y_pred_svc = grid_svc.predict(X_test_scaled)
# confusion matrix.
confusion_matrix(y_test, y_pred_svc)
# macro f1 score.
svc_macro = f1_score(y_test, y_pred_svc, average='macro')
svc_macro
# micro f1 score
svc_micro = f1_score(y_test, y_pred_svc, average='micro')
svc_micro

# Try decision tree
mdt = DecisionTreeClassifier()
param_dt = {'max_depth': [10, 100, 1e3]}
grid_dt = GridSearchCV(mdt, param_dt)
# train the model using training Dataset
grid_dt.fit(X_train_scaled, y_train)
y_pred_dt = grid_dt.predict(X_test_scaled)
# model evaluation for decision tree.
# confusion matrix.
confusion_matrix(y_test, y_pred_dt)
# macro f1 score.
dt_macro = f1_score(y_test, y_pred_dt, average='macro')
dt_macro
# micro f1 score
dt_micro = f1_score(y_test, y_pred_dt, average='micro')
dt_micro
# note that the behaviour of decision tree is identical to knn, the reason behind is unknown

# Try random RandomForestClassifier
mrf = RandomForestClassifier()
param_rf = {'max_features':('auto', 'log2', '')}
grid_rf = GridSearchCV(mrf, param_rf)
# train the model with data.
grid_rf.fit(X_train_scaled, y_train)
# model evaluation for random RandomForestClassifier
y_pred_rf = grid_rf.predict(X_test_scaled)
# confusion matrix.
confusion_matrix(y_test, y_pred_rf)
# macro f1 score.
rf_macro = f1_score(y_test, y_pred_rf, average='macro')
rf_macro
# micro f1 score
rf_micro = f1_score(y_test, y_pred_rf, average='micro')
rf_micro

# Try Gradient boost
mgb = GradientBoostingClassifier()
param_gb = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':np.arange(1, 10)}
grid_gb = GridSearchCV(mgb, param_gb)
# train the model using Dataset
grid_gb.fit(X_train_scaled, y_train)
# model evaluation for Gradient GradientBoostingClassifier
y_pred_gb = grid_gb.predict(X_test_scaled)
# confusion matrix.
confusion_matrix(y_test, y_pred_gb)
# macro f1 score.
gb_macro = f1_score(y_test, y_pred_gb, average='macro')
gb_macro
# micro f1 score
gb_micro = f1_score(y_test, y_pred_gb, average='micro')
gb_micro

# Try Naïve Bayes Classifiers
mnb = GaussianNB()
mnb.fit(X_train_scaled, y_train)
# model evaluation for Naive Bayes Classifiers
# model evaluation for Gradient GradientBoostingClassifier
y_pred_nb = mnb.predict(X_test_scaled)
# confusion matrix.
confusion_matrix(y_test, y_pred_nb)
# macro f1 score.
nb_macro = f1_score(y_test, y_pred_nb, average='macro')
nb_macro
# micro f1 score
nb_micro = f1_score(y_test, y_pred_nb, average='micro')
nb_micro

# Try Neural netwrok: one hot encoder for neural network classification
# ohe = OneHotEncoder()
# use the encoder to transform y
# y_ohe_train = ohe.fit_transform(y_train).toarray()
# y_ohe_train
# y_ohe_test = ohe.fit_transform(y_test).toarray()
# y_ohe_test
m_nn = MLPClassifier(hidden_layer_sizes=(100, 300, 300, 300, 100))
# fit the data
m_nn.fit(X_train_scaled, y_train)
# model evaluation for neural neural_network
y_pred_nn = m_nn.predict(X_test_scaled)
y_pred_nn
# encode the onehot encoded y back to original y
# confusion matrix.
confusion_matrix(y_test, y_pred_nn)
# macro f1 score.
nn_macro = f1_score(y_test, y_pred_nn, average='macro')
nn_macro
# micro f1 score
nn_micro = f1_score(y_test, y_pred_nn, average='micro')
nn_micro

# model comparison
# in terms of macro f1 score, each class has equal weight
macrof1 = plt.figure()
ax = macrof1.add_axes([10, 10, 1, 1])
model_selection = ['K Nearest Neighbors', 'Kernalized Support Vector Machines', 'Decision Trees', 'Random Forest', 'Gradient Boost', 'Naive Bayes', 'Neural network']
f1_macro_scores = [knn_macro, svc_macro, dt_macro, rf_macro, gb_macro, nb_macro, nn_macro]
ax.barh(model_selection,f1_macro_scores)
plt.title('Macro f1 scores')
plt.show()
# in terms of micro f1 score, each sample has equal weight
macrof1 = plt.figure()
ax = macrof1.add_axes([10, 10, 1, 1])
model_selection = ['K Nearest Neighbors', 'Kernalized Support Vector Machines', 'Decision Trees', 'Random Forest', 'Gradient Boost', 'Naive Bayes', 'Neural network']
f1_macro_scores = [knn_micro, svc_micro, dt_micro, rf_micro, gb_micro, nb_micro, nn_micro]
ax.barh(model_selection,f1_macro_scores)
plt.title('Micro f1 scores')
plt.show()
