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


def classification_trainers(X, y, n_repeat):
    counter = 0
    # create lists to collect the f1 scores for each model
    f1_knn_list = []
    f1_svc_list = []
    f1_dt_list = []
    f1_rf_list = []
    f1_gb_list = []
    f1_nb_list = []
    f1_knn_macro = []
    f1_svc_macro = []
    f1_dt_macro = []
    f1_rf_macro = []
    f1_gb_macro = []
    f1_nb_macro = []
    # use while loop for repeting the data splitting and training
    while counter < n_repeat:
        counter = counter + 1
        # split the training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        # scale the features.
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # we must apply the scaling to the test set that we computed for the training set
        X_test_scaled = scaler.transform(X_test)
        # train the ML model: try knn with GridSearchCV varying the number of nearest neighbour

        # knn
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
        knn_f1 = f1_score(y_test, y_pred_knn, average='micro')
        f1_knn_list.append(knn_f1)
        f1_knn_macro.append(f1_score(y_test, y_pred_knn, average='macro'))

        # try using Kernalized Support Vector Machines
        msvc = SVC()
        param_svc = {'C': np.linspace(0.1, 100), 'kernel': ('linear', 'poly', 'rbf')}
        grid_svc = GridSearchCV(msvc, param_svc)
        # train the model using training Dataset
        grid_svc.fit(X_train_scaled, y_train)
        # model evaluation for SVC
        y_pred_svc = grid_svc.predict(X_test_scaled)
        # evaluate the model by f1 scores
        svc_f1 = f1_score(y_test, y_pred_svc, average='micro')
        f1_svc_list.append(svc_f1)
        f1_svc_macro.append(f1_score(y_test, y_pred_svc, average='macro'))

        # Try decision tree
        mdt = DecisionTreeClassifier()
        param_dt = {'max_depth': [10, 100, 1e3]}
        grid_dt = GridSearchCV(mdt, param_dt)
        # train the model using training Dataset
        grid_dt.fit(X_train_scaled, y_train)
        y_pred_dt = grid_dt.predict(X_test_scaled)
        # evaluate the model by f1 scores.
        dt_f1 = f1_score(y_test, y_pred_dt, average='micro')
        f1_dt_list.append(dt_f1)
        f1_dt_macro.append(f1_score(y_test, y_pred_dt, average='macro'))

        # Try random RandomForestClassifier
        mrf = RandomForestClassifier()
        param_rf = {'max_features':('auto', 'log2', '')}
        grid_rf = GridSearchCV(mrf, param_rf)
        # train the model with data.
        grid_rf.fit(X_train_scaled, y_train)
        # model evaluation for random RandomForestClassifier
        y_pred_rf = grid_rf.predict(X_test_scaled)
        # evaluate the model by f1 scores.
        rf_f1 = f1_score(y_test, y_pred_rf, average='micro')
        f1_rf_list.append(rf_f1)
        f1_rf_macro.append(f1_score(y_test, y_pred_rf, average='macro'))

        # Try Gradient boost
        mgb = GradientBoostingClassifier()
        param_gb = {'n_estimators':[100, 500, 1e3], 'learning_rate':[0.1, 1, 10], 'max_depth':np.arange(1, 10)}
        grid_gb = GridSearchCV(mgb, param_gb)
        # train the model using Dataset
        grid_gb.fit(X_train_scaled, y_train)
        # model evaluation for Gradient GradientBoostingClassifier
        y_pred_gb = grid_gb.predict(X_test_scaled)
        # evaluate the model by f1 score.
        gb_f1 = f1_score(y_test, y_pred_gb, average='micro')
        f1_gb_list.append(gb_f1)
        f1_gb_macro.append(f1_score(y_test, y_pred_gb, average='macro'))

        # Try Naïve Bayes Classifiers
        mnb = GaussianNB()
        mnb.fit(X_train_scaled, y_train)
        # model evaluation for Naive Bayes Classifiers
        y_pred_nb = mnb.predict(X_test_scaled)
        # evaluate the model by f1 score.
        nb_f1 = f1_score(y_test, y_pred_nb, average='micro')
        f1_nb_list.append(nb_f1)
        f1_nb_macro.append(f1_score(y_test, y_pred_nb, average='macro'))

    # now for each list of f1 score, plot a box plot for each model.
    plt.figure()
    f1_scores_df = pd.DataFrame(np.transpose([f1_knn_list, f1_svc_list, f1_dt_list, f1_rf_list, f1_gb_list, f1_nb_list]), columns=['knn', 'svc', 'decision tree', 'random forest', 'gradient boost', 'naive Bayes'])
    f1_scores_df.boxplot(vert=False)
    plt.title('f1 micro score for position prediction models')
    plt.show()

    plt.figure()
    f1_scores_macro = pd.DataFrame(np.transpose([f1_knn_macro, f1_svc_macro, f1_dt_macro, f1_rf_macro, f1_gb_macro, f1_nb_macro]), columns=['knn', 'svc', 'decision tree', 'random forest', 'gradient boost', 'naive Bayes'])
    f1_scores_macro.boxplot(vert=False)
    plt.title('f1 macro score for position prediction models')
    plt.show()

    return f1_scores_df, f1_scores_macro

################################################################################
# data visualization and pre-processing
# import the data
df = pd.read_csv(r'C:\Users\sijin wang\Desktop\TOR\MLcode\ball_exercise\clean_data.csv')
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
    plt.title('box plot for ' + str(feature) + 'before removing outliers')
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
    plt.title('box plot for ' + str(feature) + 'before removing outliers')
    plt.grid(False)
    dfunique_clean.boxplot(column=[feature])
    plt.show()

# data splitting
# define the X and y: use dummy variables for y
X = dfunique_clean[['height', 'weight', 'spike', 'block']]
# y = pd.get_dummies(dfunique_clean['position_number'], columns=['position_number'])
y = dfunique_clean[['position_number']]

##################################################################################
# train and evaluate the models.
classification_trainers(X, y, 5)

# data = pd.DataFrame(np.transpose([[1, 2, 3], [4, 3, 2]]), columns=['a', 'b'])
# data.T.boxplot(vert=False)
