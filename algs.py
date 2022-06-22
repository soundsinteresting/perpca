import numpy as np
from numpy import linalg as LA
import scipy.stats as st
import statsmodels.api as sm
import os

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import cdist, euclidean


def gen_local_components(ttd=15, ini_id=2, ter_id=11, num_per_client=3, num_client=10):
    res = np.zeros((num_client, num_per_client, ttd))
    for clid in range(num_client):
        for cpid in range(num_per_client):
            res[clid, cpid, (cpid + clid) % (ter_id - ini_id + 1) + ini_id] = 1
            # res[clid, cpid, (cpid)%(ter_id-ini_id+1)+ini_id] = 1

    return res


def cluster_plot(dis, clusters):
    color = ['lightcoral', 'sienna', 'darkorange', 'greenyellow', 'seagreen',
             'aquamarine', 'cyan', 'steelblue', 'navy', 'blueviolet', 'violet', 'pink']
    N = len(dis)
    dis_sq = -dis  # *dis
    G = -0.5 * (np.eye(N) - np.ones((N, N)) / N) @ dis_sq @ (np.eye(N) - np.ones((N, N)) / N)

    evals, evecs = np.linalg.eigh(G)
    # print(evals)
    x = -evecs[:, -1] * np.sqrt(evals[-1])
    y = -evecs[:, -2] * np.sqrt(evals[-2])
    fig, ax = plt.subplots(1, 1)
    for i in range(len(x)):
        ax.scatter(x[i], y[i], color=color[clusters[i]])
    for i in range(len(dis)):
        ax.annotate(str(i % 10 + 1), (x[i], y[i]))
    plt.savefig('clietsrelation.png')


def subspace_error(U,V):
    r = len(U[0])
    pu = U@U.T
    pv = V@V.T
    return r-np.trace(pu@pv)

def subspace_error_avg(U_list,V_list):
    r = 1
    if type(U_list) == list:
        U = lambda i : U_list[i]
        r = len(U_list)
    else:
        U = lambda i : U_list
    if type(V_list) == list:
        V = lambda i : V_list[i]
        r = len(V_list)
    else:
        V = lambda i : V_list
    err = [subspace_error(U(i),V(i)) for i in range(r)]
    return np.mean(np.array(err))

def generate_data(g_cs, l_cs, d, local_ratio=0.5, num_dp=100):
    n_client = len(l_cs)
    Y = []  # np.zeros((n_client, num_dp, d))
    for i in range(n_client):
        g_dim = len(g_cs[:, 0])
        X_g = np.random.multivariate_normal(np.zeros(g_dim), np.eye(g_dim), num_dp)

        Y.append(X_g @ g_cs * (1-local_ratio))
        l_dim = len(l_cs[i, :, 0])
        X_l = np.random.multivariate_normal(np.zeros(l_dim), np.eye(l_dim), num_dp)
        Y[i] += X_l @ l_cs[i] * local_ratio
        w = np.random.multivariate_normal(np.zeros(d), 0.5 * np.identity(15), num_dp)
        Y[i] += w  # np.transpose(w)
        Y[i] = Y[i]
    return Y


def single_PCA(Yi, ngc):
    S = Yi.T @ Yi
    U, s, Vh = LA.svd(S)
    return U[:, 0:ngc]


def initial_u(Y, d, ngc, random=0):
    if random == 0:
        Ycombined = np.concatenate(Y, axis=0)
        S = Ycombined.T @ Ycombined
        evs, U = LA.eig(S)
    else:
        U = np.random.randn(d,ngc)
        U = schmit(U)
    return np.real(U[:, 0:ngc])


def optimize_U_and_Vk(Yk, Vk, Uk, Lambdak, Z, args):
    eta = args['eta']
    rho = args['rho']
    num_steps = args['local_epochs']
    S = Yk.T @ Yk
    # Optimize U and Vk
    for i in range(num_steps):
        gradu = -2 * S @ Uk
        graduadd = Lambdak + 2 * rho * (Uk - Z)
        gradv = -2 * S @ Vk
        Uk -= eta * graduadd
        Vk -= eta * gradv
        Uk, Vk = generalized_retract(Uk, Vk)
        # Uk -= eta * graduadd
    return Uk, Vk


def correctv(Yk, Vk, Uk, args):
    # Uk, Vk = generalized_retract(Uk, Vk)
    # return Uk, Vk
    #Z = schmit(np.concatenate((Uk, Vk), axis=1))
    #return Z[:, :args['ngc']], Z[:, args['ngc']:]
    Vk = Vk-Uk@Uk.T@Vk
    return Uk, generalized_retract_single(Vk,'polar')


def optimize_U_and_Vk_stiefel(Yk, Vk, Uk, args):
    eta = args['eta']
    num_steps = 1  # args['local_epochs']
    S = Yk.T @ Yk/len(Yk)
    # Optimize U and Vk
    # print("begin: sc: {}, ic: {}".format(args['ngc']-np.trace(Uk.T@Uk), np.trace(Uk.T@Vk)))
    # Uk, Vk = adjust_vk(Uk, Vk)
    # Uk, Vk = generalized_retract(Uk, Vk)
    Uk, Vk = correctv(Yk, Vk, Uk, args)
    du = len(Uk[0])
    dv = len(Vk[0])
    for i in range(num_steps):
        Wk = np.concatenate((Uk,Vk), axis=1)
        '''
        print('deviation')
        att = np.linalg.norm(Wk.T@Wk-np.eye(du+dv))
        a11 = np.linalg.norm(Uk.T@Uk-np.eye(du))
        a12 = np.linalg.norm(Uk.T@Vk)
        a22 = np.linalg.norm(Vk.T@Vk-np.eye(dv))
        print(att,a11,a12,a22)
        '''

        gradw = -2*S@Wk
        rgradw = gradw - Wk@(gradw.T@Wk+Wk.T@gradw)/2

        PRINT = False
        if PRINT:
            etar = 1/np.linalg.norm(rgradw)**2
            print('norms')
            print(np.linalg.norm(etar*gradw))
            print(np.linalg.norm(etar*rgradw))
            print('deviation')
            print(np.linalg.norm(generalized_retract_single(Wk-etar*gradw, 'polar')-generalized_retract_single(Wk-etar*gradw, 'qr')))

            print(np.linalg.norm(generalized_retract_single(Wk-etar*gradw, 'polar')-generalized_retract_single(Wk-etar*rgradw, 'polar')))
            print(np.linalg.norm(Wk-etar*rgradw-generalized_retract_single(Wk-etar*rgradw, 'polar')))

            print('===========')

        #print(np.linalg.norm(rgradw))
        #eta = 1e-7 #1/np.linalg.norm(rgradw)**2
        Wk -= eta*gradw
        #Wk -= eta*gradw
        
        #assert False
        Wk = generalized_retract_single(Wk, 'polar')
        Uk, Vk = Wk[:,:du], Wk[:,du:]
        '''
        gradu = -2 * S @ Uk
        gradv = -2 * S @ Vk
        gradu = (gradu - Uk@Uk.T@gradu - Vk@Vk.T@gradu)
        gradv = (gradv - Uk@Uk.T@gradv - Vk@Vk.T@gradv)
        Uk -= eta * (gradu)
        Vk -= eta * (gradv)
        Uk, Vk = generalized_retract(Uk, Vk, 'polar')
        '''
        # print("final: sc: {}, ic: {}".format(args['ngc'] - np.trace(Uk.T @ Uk), np.trace(Uk.T @ Vk)))

    return Uk, Vk

def optimize_U_and_Vk_stiefel_precise(Yk, Vk, Uk, args):
    eta = args['eta']
    num_steps = 1  # args['local_epochs']
    S = Yk.T @ Yk
    # Optimize U and Vk
    # print("begin: sc: {}, ic: {}".format(args['ngc']-np.trace(Uk.T@Uk), np.trace(Uk.T@Vk)))
    # Uk, Vk = adjust_vk(Uk, Vk)
    # Uk, Vk = generalized_retract(Uk, Vk)
    Uk, Vk = correctv(Yk, Vk, Uk, args)
    for i in range(num_steps):
        gradu = -2 * S @ Uk
        gradv = -2 * S @ Vk
        # gradu = (gradu - Uk@Uk.T@gradu - Vk@Vk.T@gradu)
        # gradv = (gradv - Uk@Uk.T@gradv - Vk@Vk.T@gradv)
        Uk -= eta * (gradu)
        Vk -= eta * (gradv)
        Uk, Vk = generalized_retract(Uk, Vk)
        # print("final: sc: {}, ic: {}".format(args['ngc'] - np.trace(Uk.T @ Uk), np.trace(Uk.T @ Vk)))

    return Uk, Vk


def optimize_U_and_Vk_soft(Yk, Vk, Uk, args):
    eta = args['eta']
    num_steps = 1  # args['local_epochs']
    S = Yk.T @ Yk
    # Optimize U and Vk
    print("sc: {}, ic: {}".format(args['ngc'] - np.trace(Uk.T @ Uk), np.trace(Uk.T @ Vk)))
    # Uk, Vk = adjust_vk(Uk, Vk)
    # Uk, Vk = generalized_retract(Uk, Vk)
    for i in range(num_steps):
        gradu = (-2 * S + args['lambda'] * Vk @ Vk.T) @ Uk
        gradv = (-2 * S + args['lambda'] * Uk @ Uk.T) @ Vk
        Uk -= eta * (gradu)
        Vk -= eta * (gradv)
        # Uk, Vk = generalized_retract(Uk, Vk)
        Uk = schmit(Uk)
        Vk = schmit(Vk)
    return Uk, Vk


def generalized_retract_single(Uk, method = 'polar'):
    if method == 'polar':
        u, s, vh = np.linalg.svd(Uk)
        D = np.zeros((u.shape[1], vh.shape[0]))
        for j in range(min(u.shape[1], vh.shape[0])):
            D[j, j] = 1
        reconstruct = u @ D @ vh
        return reconstruct
    elif method == 'qr':
        reconstruct = np.linalg.qr(Uk)[0]
        return reconstruct
    else:
        raise Exception('Unimplemented retraction: '+method)


def generalized_retract(Uk, Vk, method='polar'):
    du = len(Uk[0])
    dv = len(Vk[0])
    if method == 'polar':
        u, s, vh = np.linalg.svd(np.concatenate((Uk, Vk), axis=1))
        s = s / s
        # print(u.shape)
        # print(vh.shape)
        D = np.zeros((u.shape[1], vh.shape[0]))
        for j in range(min(u.shape[1], vh.shape[0])):
            D[j, j] = 1
        reconstruct = u @ D @ vh
        return reconstruct[:, :du], reconstruct[:, du:]
    elif method == 'qr':
        reconstruct = np.linalg.qr(np.concatenate((Uk, Vk), axis=1))[0]
        return reconstruct[:,:du], reconstruct[:,du:]
    else:
        raise Exception('Unimplemented retraction: '+method)


def adjust_vk(Uk, Vk):
    du = len(Uk[0])
    dv = len(Vk[0])
    q_adjusted = schmit(np.concatenate((Uk, Vk), axis=1))
    return q_adjusted[:, :du], q_adjusted[:, du:]


def single_loss(Y, U, V=None):
    m = len(Y)
    if V == None:
        v = U
    else:
        v = np.concatenate((U, V), axis=1)
    return np.linalg.norm(Y.T - v @ v.T @ Y.T, ord='fro') ** 2 / m


def loss(Y, U, V=0):
    res = 0
    k = len(Y)
    tot = 0.
    for i in range(k):
        if type(V) == int:
            v = U
        elif type(U) == list:
            Uk,Vk = adjust_vk(U[i], V[i])
            v = np.concatenate((Uk, Vk), axis=1)
        else:
            v = np.concatenate((U, V[i]), axis=1)
        m = len(Y[i])
        res += np.linalg.norm(Y[i].T - v @ v.T @ Y[i].T, ord='fro') ** 2
        tot += m
    res /= tot
    return res


def schmit(Q):
    nrow = len(Q[0])
    d = len(Q)
    for i in range(nrow):
        for j in range(i):
            Q[:, i] -= (Q[:, i] * Q[:, j]).sum() * Q[:, j]
        Q[:, i] /= np.sqrt((Q[:, i] ** 2).sum())
    return Q


def simple_eig(A, n_round=100, nc=3, non_update=0, init_Q=None):
    d = len(A)
    if init_Q.any():
        Q = init_Q
    else:
        Q = np.random.randn(d, nc)
    schmit(Q)
    for i in range(n_round):
        Q[:, non_update:] = A @ Q[:, non_update:]
        vas = [np.linalg.norm(Q[:, j]) for j in range(nc)]
        schmit(Q)
    return vas, Q


def spectral_cluster(V):
    ncl = len(V)
    afm = np.zeros((ncl, ncl))
    for i in range(ncl):
        for j in range(i):
            afm[i, j] = np.trace(V[i] @ V[i].T @ V[j] @ V[j].T)
            afm[j, i] = afm[i, j]

    maxele = np.max(afm)
    afm /= maxele
    afm = afm ** 2
    print(afm)
    print(afm[0])
    afm_copy = copy.deepcopy(afm)
    # cluster_plot(afm)
    for i in range(ncl):
        afm[i, i] -= afm[i].sum()
    clustering = SpectralClustering(n_clusters=10,
                                    assign_labels='discretize',
                                    random_state=0, affinity='precomputed').fit(afm)
    print(clustering.labels_)
    cluster_plot(afm_copy, clustering.labels_)


def personalized_pca_dgd(Y, args):
    ngc, nlc = args['ngc'], args['nlc']
    d = len(Y[0][0, :])
    num_client = args['num_client']
    rho = args['rho']

    U_init = initial_u(Y, d, ngc)
    #U_init = np.random.randn(d, ngc)
    #U_init = schmit(U_init)
    # print(U_init)
    V = [np.random.multivariate_normal(np.zeros(d), np.eye(d), nlc).T for i in range(num_client)]
    V = [schmit(Vi - U_init @ U_init.T @ Vi) for Vi in V]
    U = [copy.deepcopy(U_init) for i in range(num_client)]
    lv = []
    logpregress = False
    if 'logprogress' in args.keys():
        logpregress = True
    # spectral_cluster(V)
    for i in range(args['global_epochs']):
        # if i == 1:
        #    spectral_cluster(V)
        # 1st step
        for k in range(num_client):
            if i % 1 == 0:
                U[k], V[k] = optimize_U_and_Vk_stiefel(Y[k], V[k], U[k], args)
                # U[k], V[k] = optimize_U_and_Vk_(Y[k], V[k], U[k], args)
            else:
                U[k], V[k] = correctv(Y[k], V[k], U[k], args)
            # U[k], V[k] = optimize_U_and_Vk_soft(Y[k], V[k], U[k], args)
        # lr decay
        if i % 10 == 9:
            args['eta'] *= 1  # 0.8
        # 2nd step: avarage U and retract
        U_avg = sum(U[k] for k in range(num_client)) / num_client
        U_avg = generalized_retract_single(U_avg,'qr')

        # 3rd step: broadcast U
        for k in range(num_client):
            U[k] = copy.deepcopy(U_avg)
            # U[k], V[k] = generalized_retract(U_avg, V[k])

        # 4-th step: adjust V
        # for k in range(num_client):
        #    U[k] , V[k] = adjust_vk(U[k], V[k])
        ls = loss(Y, U, V)
        if logpregress:
            print("[{}/{}]: loss {}".format(i, args['global_epochs'], ls))
        lv.append(ls)

    # spectral_cluster(V)
    for k in range(num_client):
        U[k] , V[k] = adjust_vk(U[k], V[k])
    return U, V, lv

def two_shot_pca(Y, args):
    ngc, nlc = args['ngc'], args['nlc']
    d = len(Y[0][0, :])
    num_client = args['num_client']
    
    U1 = [np.zeros((d,ngc+nlc)) for i in range(num_client)]
    V = []
    lv = []

    logpregress = False
    if 'logprogress' in args.keys():
        logpregress = True

    # calulate pc of each client
    for k in range(num_client):
        U1[k] = single_PCA(Y[k], ngc+nlc)
        
    # server calculates the aggregations of pcs
    U_aggregate = np.concatenate(U1, axis=1)
    U2 = single_PCA(U_aggregate.T,ngc)
    #print('u2 shape')
    #print(U2.shape)
    #print(U2)

    # calculates the local pcs by deflation
    for k in range(num_client):
        #print('shapes')
        #print(Y[k].shape)
        #print(U2.shape)
        
        V.append(single_PCA(Y[k]-Y[k]@U2@U2.T, nlc))
        #print(V[k].shape)
    #print('V[0] shape')
    #print(V[0].shape)
    lv = []
    return [U2 for i in range(len(V))], V, lv

def personalized_pca_admm(Y, args):
    ngc, nlc = args['ngc'], args['nlc']
    d = len(Y[0, 0, :])
    num_client = args['num_client']
    rho = args['rho']

    U_init = initial_u(Y, d, ngc)
    V = [np.random.multivariate_normal(np.zeros(d), np.eye(d), nlc).T for i in range(num_client)]
    V = [schmit(Vi - U_init @ U_init.T @ Vi) for Vi in V]
    U = [copy.deepcopy(U_init) for i in range(num_client)]
    Lambda = [np.zeros((d, ngc)) for i in range(num_client)]
    Z = copy.deepcopy(U_init)
    # spectral_cluster(V)
    for i in range(args['global_epochs']):
        # if i == 1:
        #    spectral_cluster(V)
        # 1st step
        for k in range(num_client):
            U[k], V[k] = optimize_U_and_Vk(Y[k], V[k], U[k], Lambda[k], Z, args)

        # 2nd step: avarage Z
        Z = sum(U[k] + Lambda[k] / rho for k in range(num_client)) / num_client
        Z = generalized_retract_single(Z)

        dl = sum(np.linalg.norm(Z - U[k]) ** 2 for k in range(num_client)) / num_client
        # 3rd step: updata Lambda:
        for k in range(num_client):
            Lambda[k] += 2 * rho * (U[k] - Z)
            Lambda[k] *= 0
        print("[{}/{}]: loss {}, dev loss {}".format(i, args['global_epochs'], loss(Y, Z, V), np.sqrt(dl)))

    # spectral_cluster(V)
    # return U,V
    return [Z for i in range(len(U))], V

def logistic_regression_single(Xtrain,ytrain,Xtest,ytest):
    from sklearn.linear_model import LogisticRegression
    #print('logistic regression')
    #print(Xtrain.shape)
    #print(ytrain.shape)
    clf = LogisticRegression(random_state=0).fit(Xtrain.T, ytrain)
    ytrainpred = clf.predict(Xtrain.T)
    trainacc = np.sum(ytrainpred==ytrain)/len(ytrain)
    ytestpred = clf.predict(Xtest.T)
    testacc = np.sum(ytestpred==ytest)/len(ytest)
    return trainacc, testacc

def logistic_regression(Xtrains,ytrains,Xtests,ytests):
    trainaccs = []
    testaccs = []
    for i in range(len(Xtrains)):
        tracc,tsacc = logistic_regression_single(Xtrains[i],ytrains[i],Xtests[i],ytests[i])
        trainaccs.append(tracc)
        testaccs.append(tsacc)
    trainaccs = np.array(trainaccs)
    testaccs = np.array(testaccs)
    return np.mean(trainaccs), np.mean(testaccs)

if __name__ == "__main__":
    d = 10
    np.random.seed(0)
    A = np.random.randn(d,d)
    Y = np.random.randn(d,d)
    print(np.linalg.norm(A-Y))
    U,s,Vt = np.linalg.svd(Y*100+A*0)
    U = U[:,:2]
    S = A.T@A
    nabla = S@U
    grad = nabla - U@(U.T@nabla+nabla.T@U)/2
    maxeta = 1000
    mineta = 0.001
    ratio = maxeta/mineta
    tts = 100
    etas = []
    errs = []
    err2s = []
    err3s = []
    for i in range(tts):
        eta = mineta * (ratio**(i/tts))
        etas.append(eta)
        Udrpolar = generalized_retract_single(U+eta*nabla)
        Upolar = generalized_retract_single(U+eta*grad)
        err = np.linalg.norm(Udrpolar-Upolar)
        err2 = np.linalg.norm(U+eta*grad-Upolar)
        err3 = np.linalg.norm(Ud)
        errs.append(err)
        err2s.append(err2)
        err3s.append(err3)

    plt.loglog(etas, errs,label='error')
    plt.loglog(etas, err2s, label='error2')
    plt.loglog(etas, err3s, label='error3')

    plt.legend()
    plt.savefig('stiefelretrac.png')