"""
    Using MLP Regression for model fitting
"""

# Author: Shruti Vyas

import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import time, os
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

# data_file = "./data/coked_mo_2step.txt"
# data_file = "./data/coked_mo_control.txt"
# out_file = "./output/Lreg_cokedCont.txt"

in_dir = "./data/"
out_dir = "./output/"

np.seterr(all='warn')

ACTIVATION_TYPES = ["logistic", "tanh", "relu"]

def load_data(data_file):
    data = np.loadtxt(data_file, skiprows=1)
    
    #print data.size
    #print data.shape
    
    # this function returns two values
    # 1. input features of the model to train: 
    #       from first column to second last
    #2. output data (percentage leaching) for the given training data:
    #       from last column

    return data[:,0:-1], data[:,-1:].ravel()
    
def perform_linear_reg(input_file, out_dir, data_file):

    out_file = out_dir + data_file
    
    # load data
    X, y = load_data(input_file)
    print X.shape, y.shape
    # Create linear regression object
    X = StandardScaler().fit_transform(X)
    # fit to data then transform
    regr = LinearRegression(fit_intercept=True, normalize=False)

    # Train the model using the training sets
    regr.fit(X, y)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X) - y) ** 2))
      # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X, y))
    print regr.intercept_
    
    fp = open(out_file, 'w')
    fp.write('Coefficients\n')    
    np.savetxt(fp, regr.coef_, fmt='%.10f')
    
    fp.write('\nIntercept \n %.10f\n' % (regr.intercept_))
    
    # print >>fp, 'Coefficients: \n', regr.coef_
    # fp.write("Residual sum of squares: %.2f", np.mean((regr.predict(X) - y) ** 2)
    # fp.write('Variance score: %.2f' % regr.score(X, y)
    
    fp.close()

def perform_regression(input_file, data_file):

    L1 = 1,10,1
    L2 = 1,10,1
    L3 = 1,10,1

    # regularization
    alpha=0.001
    solver='lbfgs'

    global out_dir

    out_dir = out_dir + str(L1[1]) + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f_name = os.path.splitext(data_file)[0]

    f_name = f_name + time.strftime("_%m_%d_%H_%M_%S") + '_L1_' + str(L1[0]) + '_' + str(L1[1]) + '_L2_' + str(L2[0]) + '_' + str(L2[1]) + '_L3_' + str(L3[0]) + '_' + str(L3[1]) + '.txt'

    out_file = out_dir + f_name

    fp = open(out_file, 'w')

    # load data
    X, y = load_data(input_file)
    print X.shape, y.shape

    scaler = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1,1))

    X = scaler.transform(X)
    # y = scaler_y.transform(y.reshape(-1,1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    best_mlp = None
    best_mlp_l = None
    best_mlp_r = None
    best_mlp_t = None
    best_score = 0.0
    best_score_l = 0.0
    best_score_r = 0.0
    best_score_t = 0.0
    best_network = None
    best_activation = None
    
    # this is a multi line comment
    """
    if 0:
        print 'testing....'
        print 'comment is on'
    """

    # for single layer
    # for three different types of activation function run a for loop
    # we are using tanh, relu and singmoid
    # this loop will run three times for each of these activation fcuntions
    for activation in ACTIVATION_TYPES:
        f_name_1 = os.path.splitext(data_file)[0]
        f_name_1 = f_name_1 + '_' + activation + '_L1.txt'
        out_file_1 = out_dir + f_name_1
        fp1 = open(out_file_1, 'w')
        # Now we will start with the first layer
        # we will test for a range like 5-10...
        for l1 in range(L1[0],L1[1],L1[2]):
            t0 = time.time()
            mlp = MLPRegressor(solver=solver, hidden_layer_sizes=(l1,),
                        max_iter=200, shuffle=True, random_state=1,
                        activation=activation, alpha=alpha)
                           
            scores = cross_val_score(mlp, X_train, y_train, cv=10)
            fp.write("Activation: %s, Network : (%d)\n" % (activation, l1))
            fp.write("Accuracy: %0.4f (+/- %0.4f)\n" % (scores.mean(), scores.std()))
            print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
            fp1.write("%0.6f\t%d\n" %(scores.mean(), l1))
            score = np.mean(scores)
            print 'time : ', time.time() - t0
            if score > best_score:
                print score
                best_score = score
                best_mlp = mlp
                best_network = (l1,)
                best_activation = activation
            if activation == 'logistic' and score > best_score_l:
                best_score_l = score
                best_mlp_l = mlp
            if activation == 'relu' and score > best_score_r:
                best_score_r = score
                best_mlp_r = mlp
            if activation == 'tanh' and score > best_score_t:
                best_score_t = score
                best_mlp_t = mlp
        fp1.close()

    for activation in ACTIVATION_TYPES:
        f_name_1 = os.path.splitext(data_file)[0]
        f_name_1 = f_name_1 + '_' + activation + '_L2.txt'
        out_file_1 = out_dir + f_name_1
        fp1 = open(out_file_1, 'w')
        for l1 in range(L1[0],L1[1],L1[2]):
        # for l1 in range(L1[0],L1[1]):
            for l2 in range(L2[0], L2[1], L2[2]):
                
                t0 = time.time()
                mlp = MLPRegressor(solver=solver, hidden_layer_sizes=(l1,l2,),
                           max_iter=200, shuffle=True, random_state=1,
                           activation=activation, alpha=alpha)
                           
                scores = cross_val_score(mlp, X_train, y_train, cv=10)
                print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
                fp.write("Activation: %s, Network : (%d, %d)\n" % (activation, l1, l2))
                fp.write("Accuracy: %0.4f (+/- %0.4f)\n" % (scores.mean(), scores.std()))
                fp1.write("%0.6f\t%d\t%d\n" %(scores.mean(), l1, l2))
                print 'time : ', time.time() - t0
                score = np.mean(scores)
                if score > best_score:
                    print score
                    best_score = score
                    best_mlp = mlp
                    best_network = (l1,l2)
                    best_activation = activation
                if activation == 'logistic' and score > best_score_l:
                    best_score_l = score
                    best_mlp_l = mlp
                if activation == 'relu' and score > best_score_r:
                    best_score_r = score
                    best_mlp_r = mlp
                if activation == 'tanh' and score > best_score_t:
                    best_score_t = score
                    best_mlp_t = mlp
        fp1.close()

    for activation in ACTIVATION_TYPES:
        f_name_1 = os.path.splitext(data_file)[0]
        f_name_1 = f_name_1 + '_' + activation + '_L3.txt'
        out_file_1 = out_dir + f_name_1
        fp1 = open(out_file_1, 'w')
        for l1 in range(L1[0],L1[1],L1[2]):
            # for l2 in range(L2[0], L2[1]):
            for l2 in range(L2[0], L2[1], L2[2]):
                for l3 in range(L3[0], L3[1], L3[2]):
                
                    t0 = time.time()
                    mlp = MLPRegressor(solver=solver, hidden_layer_sizes=(l1,l2,l3,),
                           max_iter=200, shuffle=True, random_state=1,
                           activation=activation, alpha=alpha)
                           
                    scores = cross_val_score(mlp, X_train, y_train, cv=10)
                    print 'time : ', time.time() - t0
                    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
                    fp.write("Activation: %s, Network : (%d, %d, %d)\n" % (activation, l1, l2, l3))
                    fp.write("Accuracy: %0.4f (+/- %0.4f)\n" % (scores.mean(), scores.std()))
                    fp1.write("%0.6f\t%d\t%d\t%d\n" %(scores.mean(), l1, l2, l3))

                    score = np.mean(scores)
                    if score > best_score:
                        print score
                        best_score = score
                        best_mlp = mlp
                        best_network = (l1,l2,l3)
                        best_activation = activation
                    if activation == 'logistic' and score > best_score_l:
                        best_score_l = score
                        best_mlp_l = mlp
                    if activation == 'relu' and score > best_score_r:
                        best_score_r = score
                        best_mlp_r = mlp
                    if activation == 'tanh' and score > best_score_t:
                        best_score_t = score
                        best_mlp_t = mlp
                
        fp1.close()
    best_mlp.fit(X_train, y_train)
    test_score = best_mlp.score(X_test, y_test)
    print 'Best Model : ', best_mlp.n_layers_
    print 'Best Network : ', best_network
    print 'Best activation : ', best_activation
    print 'Best score : ', best_score
    print 'Test score : ', test_score
    print 'biases:', best_mlp.intercepts_ 
    print 'coefficients:', best_mlp.coefs_ 

    best_mlp_l.fit(X_train, y_train)
    print 'Activation : logistic'
    print 'Best Model : ', best_mlp_l.n_layers_
    print 'Best score : ', best_score_l
    print 'Test score : ', best_mlp_l.score(X_test, y_test)

    best_mlp_r.fit(X_train, y_train)
    print 'Activation : relu'
    print 'Best Model : ', best_mlp_r.n_layers_
    print 'Best score : ', best_score_r
    print 'Test score : ', best_mlp_r.score(X_test, y_test)

    best_mlp_t.fit(X_train, y_train)
    print 'Activation : tanh'
    print 'Best Model : ', best_mlp_t.n_layers_
    print 'Best score : ', best_score_t
    print 'Test score : ', best_mlp_t.score(X_test, y_test)

    fp.write('Best Model, Number of layers : %d\n' %(best_mlp.n_layers_))
    fp.write('Best Network : ')
    for l in best_network:
        fp.write(' %d ' %(l))
    fp.write('\nBest activation : %s\n' %( best_activation))
    fp.write('Best score : %f\n' %( best_score))
    fp.write('Test score : %f\n' %( test_score))
    fp.write('Biases : \n')
    for l in best_mlp.intercepts_:
        fp.write('Layer : ')
        for b in l:
            fp.write(' %.6f ' %(b))
        fp.write('\n')
    fp.write('Coefficients\n')
    for l in best_mlp.coefs_:
        fp.write('Layer : \n')
        np.savetxt(fp, l, fmt='%.6f')
        fp.write('\n')

    fp.write('\nActivation : %s\n' %( 'logistic'))
    fp.write('Best Model, Number of layers : %d\n' %(best_mlp_l.n_layers_))
    fp.write('Best score : %f\n' %( best_score_l))
    fp.write('Test score : %f\n' %( best_mlp_l.score(X_test, y_test)))

    fp.write('\nActivation : %s\n' %( 'relu'))
    fp.write('Best Model, Number of layers : %d\n' %(best_mlp_r.n_layers_))
    fp.write('Best score : %f\n' %( best_score_r))
    fp.write('Test score : %f\n' %( best_mlp_r.score(X_test, y_test)))

    fp.write('\nActivation : %s\n' %( 'tanh'))
    fp.write('Best Model, Number of layers : %d\n' %(best_mlp_t.n_layers_))
    fp.write('Best score : %f\n' %( best_score_t))
    fp.write('Test score : %f\n' %( best_mlp_t.score(X_test, y_test)))

    fp.close()

    # save the trained best fit model
    spath = out_dir + '/mlp/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    spath = spath + 'mlp.pkl'
    joblib.dump(best_mlp, spath)

    spath = out_dir + '/mlp_l/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    spath = spath + 'mlp.pkl'
    joblib.dump(best_mlp_l, spath)

    spath = out_dir + '/mlp_r/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    spath = spath + 'mlp.pkl'
    joblib.dump(best_mlp_r, spath)

    spath = out_dir + '/mlp_t/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    spath = spath + 'mlp.pkl'
    joblib.dump(best_mlp_t, spath)

    spath = out_dir + '/scaler/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    scaler_path = spath + '/scaler.pkl'
    joblib.dump(scaler, scaler_path)

    spath = out_dir + '/scaler_y/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    scaler_path = spath + '/scaler.pkl'
    joblib.dump(scaler_y, scaler_path)

    # dump the test train data
    spath = out_dir + '/data/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    x_train_path = spath + '/x_train.txt'
    np.savetxt(x_train_path, X_train)
    x_test_path = spath + '/x_test.txt'
    np.savetxt(x_test_path, X_test)
    y_train_path = spath + '/y_train.txt'
    np.savetxt(y_train_path, y_train)
    y_test_path = spath + '/y_test.txt'
    np.savetxt(y_test_path, y_test)

    days = np.arange(31)
    pd = np.ones(31)
    ps = np.ones(31)*40
    temp = np.ones(31)*37

    d = np.vstack([days, pd, ps, temp]).transpose()
    
    d = scaler.transform(d)

    pred = best_mlp_l.predict(d)
    # pred = scaler_y.inverse_transform(pred)
    plt.plot(days, pred, label=activation)
    # plt.show()
    f_name = os.path.splitext(data_file)[0]

    f_name = f_name + time.strftime("_%m_%d_%H_%M_%S") + '_L1_' + str(L1[0]) + '_' + str(L1[1]) + '_L2_' + str(L2[0]) + '_' + str(L2[1]) + '_L3_' + str(L3[0]) + '_' + str(L3[1]) + '.png'

    out_file = out_dir + f_name
    plt.savefig(out_file, dpi=80)

    
# this is the main entry point of the program
if __name__ == '__main__':

    if (len(sys.argv) != 2):
        # print sys.argv
        # print len(sys.argv)
        print "usage : python ", sys.argv[0], 'data_file_path'
        # exit(0)
    
    input_file = sys.argv[1]
    # input_file = data_file

    data_file = os.path.split(input_file)[1]

    perform_regression(input_file, data_file)
    
    #perform_linear_reg(input_file, out_dir, data_file)


