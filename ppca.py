import numpy as np
from numpy import linalg as LA
import scipy.stats as st
import statsmodels.api as sm
import os

import matplotlib.pyplot as plt
import copy
from algs import *

def toytest():
    args = {
        'method':'power',
        'd':15,
        'num_client':100,
        'nlc':5,
        'ngc':2,
        'num_dp_per_client':100,
        'global_epochs':30,
        'local_epochs':10,
        'n_power':1,
        'lr':0.001,
        'eta':0.1,
        'rho':1,
        'decay':1-0.05,
    }

    #num_client=20
    np.random.seed(2021)
    lcs = gen_local_components(ttd=args['d'], ini_id=2, ter_id=11, num_per_client=args['nlc'], num_client=args['num_client'])
    gcs = np.array([[1/np.sqrt(2.), 1/np.sqrt(2.), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/np.sqrt(3.), 1/np.sqrt(3.), 1/np.sqrt(3.)]])
    #print(lcs.shape)
    gcs/=10
    print(lcs[0])
    Y=generate_data(g_cs=gcs,l_cs=lcs,d=args['d'],num_dp=args['num_dp_per_client'])
    #print(Y)

    U_glb = initial_u(Y, d=args['d'], ngc=args['nlc']+args['ngc'])
    print('statistical optimal training loss')
    print(loss(Y, gcs.T, [lc.T for lc in lcs]))
    #print(U_glb)
    #U = initial_u(Y, d=args['d'], ngc=args['ngc'])
    #print(U)
    Y_test = generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['num_dp_per_client'])
    print('global model test loss:')
    print(loss(Y_test, U_glb))
    for method in ['power']:
        print(method)
        args['method']=method
        U, V, lv = personalized_pca_dgd(Y, args=args)
        #U, V = personalized_pca_admm(Y, args=args)
        print('personalized model test loss:')
        #print(Y_test.shape)
        #print(U_p.shape)
        #print(V[0].shape)
        print(loss(Y_test, U, V))
        #print(U[0])
        #print(V[0][:,0])
        #print(V[0][:,1])
        #print(V[0][:,2])
        #print(U[1])

    #v = pca_by_gd(Y[0],5,0.01,100)
    #print(v)

    #evs, newU = LA.eig(Y[0].T @ Y[0])
    #print(newU[:,0:5])
    #projected = pca(np.transpose(Y[0]), 5)
    #print(projected)

def borrowpowertest():
    args = {
        'method':'power',
        'd':15,
        'num_client':100,
        'nlc':5,
        'ngc':4,
        'num_dp_per_client':100,
        'global_epochs':30,
        'local_epochs':10,
        'n_power':1,
        'lr':0.001,
        'eta':0.1,
        'rho':1,
        'decay':1-0.05,
    }

    #num_client=20
    np.random.seed(2021)
    lcs = gen_local_components(ttd=args['d'], ini_id=2, ter_id=11, num_per_client=args['nlc'], num_client=args['num_client'])
    gcs = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    #print(lcs.shape)
    #gcs/=10
    #print(lcs[0])
    Y1=generate_data(g_cs=gcs,l_cs=lcs[:len(lcs)//2],d=args['d'],num_dp=args['num_dp_per_client']//10)
    Y2=generate_data(g_cs=gcs,l_cs=lcs[len(lcs)//2:],d=args['d'],num_dp=args['num_dp_per_client'])
    #print(Y1.shape)
    #print(Y2.shape)
    #Y = np.concatenate((Y1,Y2), axis=0)
    Y = Y1+Y2
    lpcs = [single_PCA(Yi, args['nlc']+args['ngc']) for Yi in Y]
    singletrainingloss = np.array([single_loss(Y[i], lpcs[i]) for i in range(len(Y))])
    U_glb = initial_u(Y, d=args['d'], ngc=args['nlc']+args['ngc'])
    print('statistical optimal training loss')
    print(loss(Y, gcs.T, [lc.T for lc in lcs]))
    #print(U_glb)
    #U = initial_u(Y, d=args['d'], ngc=args['ngc'])
    #print(U)
    Y_test = generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['num_dp_per_client'])
    print('global model test loss:')
    print(loss(Y_test, U_glb))
    print('indiv model training loss:')
    print(np.mean(singletrainingloss[:len(singletrainingloss)//2]), np.mean(singletrainingloss[len(singletrainingloss)//2:]))

    print('indiv model test loss:')
    singletestloss = np.array([single_loss(Y_test[i], lpcs[i]) for i in range(len(Y_test))])
    print(np.mean(singletestloss[:len(singletestloss)//2]), np.mean(singletestloss[len(singletestloss)//2:]))

    for method in ['power']:
        print(method)
        args['method']=method
        U, V, lv = personalized_pca_dgd(Y, args=args)
        #U, V = personalized_pca_admm(Y, args=args)
        print('personalized model test loss:')
        #print(Y_test.shape)
        #print(U_p.shape)
        #print(V[0].shape)
        print(loss(Y_test, U, V))
        #print(U[0])
        #print(V[0][:,0])
        #print(V[0][:,1])
        #print(V[0][:,2])
        #print(U[1])

    #v = pca_by_gd(Y[0],5,0.01,100)
    #print(v)

    #evs, newU = LA.eig(Y[0].T @ Y[0])
    #print(newU[:,0:5])
    #projected = pca(np.transpose(Y[0]), 5)
    #print(projected)

def img_test():
    args = {
        'method': 'power',
        'd': 100,
        'num_client': 4,
        'nlc': 30,
        'ngc': 30,
        'num_dp_per_client': 100,
        'global_epochs': 100,
        'local_epochs': 1,
        'n_power': 1,
        'lr': 0.01,
        'eta': 0.001,
        'rho': 100,
        'lambda': 0,
        'decay': 1 - 0.1,
    }

    # num_client=20
    np.random.seed(2021)
    from imgpro import gen_img_data
    Y = gen_img_data()
    # print(Y)
    args['num_client'] = len(Y)
    args['d'] = len(Y[0,0])
    args['num_dp_per_client'] = len(Y[0])
    U_glb = initial_u(Y, d=args['d'], ngc=args['nlc'] + args['ngc'])
    print(U_glb.shape)
    reconstruct0 = (U_glb @ U_glb.T @ (Y[0].T)).T
    plt.imshow(reconstruct0)
    plt.axis('off')
    plt.show()
    # print(U_glb)
    # U = initial_u(Y, d=args['d'], ngc=args['ngc'])
    # print(U)
    '''
    for figidx in range(len(Y)):
        # print(loss(Y_test, U, V))
        # figidx = 3
        reconstruct0 = Y[figidx]
        plt.imshow(reconstruct0, cmap='gray')
        plt.axis('off')
        plt.savefig('video/catscat/' + 'combined_' + str(figidx) + '.png')
    '''
    Y_test = copy.deepcopy(Y) # generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['num_dp_per_client'])
    print('global model test loss:')
    print(loss(Y_test, U_glb))
    for method in ['power']:
        print(method)
        args['method'] = method
        '''
        U, V, lv = personalized_pca_dgd(Y, args=args)
        lv = np.array(lv)
        loglv = np.log(1e-20+lv-lv[-1])/np.log(10)
        plt.plot(range(len(lv)), loglv)
        plt.savefig('training_loss.png')
        '''
        args['rho'] = 1e7
        args['local_epochs']=10
        #args['']
        U, V = personalized_pca_admm(Y, args=args)
        print('personalized model test loss:')
        # print(Y_test.shape)
        # print(U_p.shape)
        # print(V[0].shape)

        for figidx in range(len(Y)):
            #print(loss(Y_test, U, V))
            #figidx = 3
            print('saving image {}'.format(figidx))
            Ui, Vi = consensus(None,V[figidx], U[figidx], args)
            reconstruct0 = (Vi@Vi.T@Y[figidx].T).T
            plt.imshow(reconstruct0,cmap='gray')
            plt.axis('off')
            plt.savefig('video/catscat/'+'cat_'+str(figidx)+'.png')

            reconstruct0 = (Ui@Ui.T @ Y[figidx].T).T
            plt.imshow(reconstruct0,cmap='gray')
            plt.axis('off')
            #plt.show()
            plt.savefig('video/catscat/'+'bg_'+str(figidx)+'.png')

def intro_example():
    args = {
        'method':'power',
        'd':3,
        'num_client':2,
        'nlc':1,
        'ngc':1,
        'num_dp_per_client':100,
        'global_epochs':300,
        'local_epochs':10,
        'n_power':1,
        'lr':0.001,
        'eta':0.01,
        'rho':1,
        'decay':1-0.05,
    }

    #num_client=20
    np.random.seed(2021)
    gcs = np.array([[0,0,1]])
    theta = 30/180*np.pi
    lcs = np.array([[[np.cos(theta/2),np.sin(theta/2),0]],[[np.cos(theta/2),-np.sin(theta/2),0]]])
    gsigma = 1
    lsigma = 2
    theta1 = np.random.rand(args['num_dp_per_client'])*2*np.pi
    theta1 = theta1.reshape(len(theta1),1)
    # Y has dimension (n_client, num_dp, d)
    Y1 = gsigma*np.cos(theta1)*gcs+lsigma*np.sin(theta1)*lcs[0]
    theta2 = np.random.rand(args['num_dp_per_client'])*2*np.pi
    theta2 = theta1.reshape(len(theta2),1)

    Y2 = gsigma*np.cos(theta2)*gcs+lsigma*np.sin(theta2)*lcs[1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Y1[:,0], Y1[:,1], Y1[:,2], color='pink')
    ax.scatter(Y2[:,0], Y2[:,1], Y2[:,2], color='blue',alpha=0.5)
    Y = np.stack((Y1,Y2))
    U_glb = initial_u(Y, d=args['d'], ngc=args['nlc']+args['ngc'])
    UNIFORM = True
    scale = 2.1
    U_glb *= scale
    if UNIFORM:
        for i in range(2):
            if U_glb[0, 0] < 0:
                U_glb *= -1
            ax.quiver(
                    0, 0, 0,  # <-- starting point of vector
                    U_glb[0,i], U_glb[1,i], U_glb[2,i],  # <-- directions of vector
                    color='black', alpha=1., lw=2, label='global component {}'.format(i+1),
                )
    else:
        U, V, lv = personalized_pca_dgd(Y, args=args)
        U = [ui*scale for ui in U]
        V = [vi*scale for vi in V]
        #print(U)
        if U[0][0,0]<0:
            U[0] *= -1
        ax.quiver(
                0, 0, 0,  # <-- starting point of vector
                U[0][0,0], U[0][1,0], U[0][2,0],  # <-- directions of vector
                color='black', alpha=1., lw=2, label='global component'
            )
        if V[0][0,0]<0:
            V[0] *= -1
        ax.quiver(
            0, 0, 0,  # <-- starting point of vector
            V[0][0, 0], V[0][1, 0], V[0][2, 0],  # <-- directions of vector
            color='pink', alpha=1., lw=2, label='dataset 1\'s local component',
        )
        if V[1][0,0]<0:
            V[1] *= -1
        ax.quiver(
            0, 0, 0,  # <-- starting point of vector
            V[1][0, 0], V[1][1, 0], V[1][2, 0],  # <-- directions of vector
            color='blue', alpha=1., lw=2, label='dataset 2\'s local component',
        )
    ax.set_xlim3d(-2.5, 2.5)
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_zlim3d(-2.01, 2.01)
    ax.view_init(40, -50)
    plt.legend(prop={'size': 14})
    plt.savefig('intro_example_ppca_{}.png'.format(not UNIFORM),bbox='tight')

    '''
    

    U_glb = initial_u(Y, d=args['d'], ngc=args['nlc']+args['ngc'])
    print('statistical optimal training loss')
    print(loss(Y, gcs.T, [lc.T for lc in lcs]))
    #print(U_glb)
    #U = initial_u(Y, d=args['d'], ngc=args['ngc'])
    #print(U)
    Y_test = generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['num_dp_per_client'])
    print('global model test loss:')
    print(loss(Y_test, U_glb))
    for method in ['power']:
        print(method)
        args['method']=method
        U, V, lv = personalized_pca_dgd(Y, args=args)
        #U, V = personalized_pca_admm(Y, args=args)
        print('personalized model test loss:')
        #print(Y_test.shape)
        #print(U_p.shape)
        #print(V[0].shape)
        print(loss(Y_test, U, V))
        #print(U[0])
        #print(V[0][:,0])
        #print(V[0][:,1])
        #print(V[0][:,2])
        #print(U[1])

    #v = pca_by_gd(Y[0],5,0.01,100)
    #print(v)

    #evs, newU = LA.eig(Y[0].T @ Y[0])
    #print(newU[:,0:5])
    #projected = pca(np.transpose(Y[0]), 5)
    #print(projected)
    '''


def toy_example1():
    import json
    args = {
        'method': 'power',
        'd': 3,
        'num_client': 2,
        'nlc': 1,
        'ngc': 1,
        'num_dp_per_client': 100,
        'global_epochs': 100,
        'local_epochs': 10,
        'n_power': 1,
        'lr': 0.001,
        'eta': 0.01,
        'rho': 1,
        'decay': 1 - 0.05,
    }

    # num_client=20
    np.random.seed(2021)
    gcs = np.array([[0, 0, 1]])
    resdict = {}
    for alpha in np.linspace(1,90,100):
        resdict[alpha] = []
        for number in range(5):
            theta = alpha / 180 * np.pi
            lcs = np.array([[[np.cos(theta / 2), np.sin(theta / 2), 0]], [[np.cos(theta / 2), -np.sin(theta / 2), 0]]])
            gsigma = 1
            lsigma = 2
            theta1 = np.random.rand(args['num_dp_per_client']) * 2 * np.pi
            theta1 = theta1.reshape(len(theta1), 1)
            # Y has dimension (n_client, num_dp, d)
            Y1 = gsigma * np.cos(theta1) * gcs + lsigma * np.sin(theta1) * lcs[0]
            theta2 = np.random.rand(args['num_dp_per_client']) * 2 * np.pi
            theta2 = theta1.reshape(len(theta2), 1)

            Y2 = gsigma * np.cos(theta2) * gcs + lsigma * np.sin(theta2) * lcs[1]
            Y = np.stack((Y1, Y2))
            U_glb = initial_u(Y, d=args['d'], ngc=args['nlc'] + args['ngc'])
            scale = 2.1
            U_glb *= scale

            U, V, lv = personalized_pca_dgd(Y, args=args)
            resdict[alpha].append(lv)
        resdict[alpha] = np.stack(resdict[alpha])

    # create json object from dictionary
    #json = json.dumps(resdict)

    # open file for writing, "w"
    #f = open("dict.json", "w")

    # write json object to file
    #f.write(json)

    # close file
    #f.close()
    tv = []
    lv = []
    for alpha in resdict:
        ar = alpha / 180 * np.pi
        theta = 1-np.cos(ar/2)
        tv.append(theta)
        lv.append(np.log(np.mean(resdict[alpha], axis=0))[-1]/np.log(10))
        #plt.plot(range(len(resdict[alpha][0])), np.log(np.mean(resdict[alpha], axis=0)),label=alpha)
    plt.plot(tv, lv)
    plt.xlabel(r'$\theta$',fontsize=20)
    plt.ylabel('Final log-error',fontsize=20)
    #plt.legend()
    plt.savefig('logerrortotheta.png')

    #(lv)


if __name__ == "__main__":
    borrowpowertest()