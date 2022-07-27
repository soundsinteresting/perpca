import numpy as np
from numpy import linalg as LA
import scipy.stats as st
import statsmodels.api as sm
import os

import matplotlib.pyplot as plt
import copy
from algs import *

class Experiment():
    def toytest(inputargs):
        args = {
            'method':'power',
            'd':15,
            'num_client':100,
            'nlc':10,
            'ngc':2,
            'num_dp_per_client':1000,
            'global_epochs':30,
            'local_epochs':10,
            'n_power':1,
            'eta':0.1,
            'rho':1,
            'decay':1-0.05,
        }

        #num_client=20
        np.random.seed(2022)
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
        print('statistical optimal test loss')
        print(loss(Y_test, gcs.T, [lc.T for lc in lcs]))
        
        print('global model test loss:')
        print(loss(Y_test, U_glb))

        U, V, lv = personalized_pca_dgd(Y, args=args)

        print('personalized model test loss:')     
        print(loss(Y_test, U, V))


        U, V, lv = two_shot_pca(Y, args=args)
        print('two shot train loss:')     
        print(loss(Y, U, V))

        print('two shot test loss:')     
        print(loss(Y_test, U, V))



        #v = pca_by_gd(Y[0],5,0.01,100)
        #print(v)

        #evs, newU = LA.eig(Y[0].T @ Y[0])
        #print(newU[:,0:5])
        #projected = pca(np.transpose(Y[0]), 5)
        #print(projected)

    def borrowpowertest(inputargs):
        args = {
            'd':15,
            'num_client':100,
            'nlc':1,
            'ngc':1,
            'seed': 2021,
            'local_ratio':0.99,
            'num_dp_per_client':10000,
            'test_num_dp_per_client':100,
            'global_epochs':100,
            'local_epochs':10,
            'choice1':1,
            #'adaptivestepsize':1,
            #'logprogress':1,
            'n_power':1,
            'eta':1e-1,#0.01,
            'rho':1,
            'decay':1-0.05,
            #'aggregationinit':1,
            #'randominit':1,
        }
        for key in inputargs:
            args[key] = inputargs[key]
        print(args)
        #num_client=20
        #np.random.seed(2021)
        np.random.seed(args['seed'])
        lcs = gen_local_components(ttd=args['d'], ini_id=args['ngc'], ter_id=args['d']-1, num_per_client=args['nlc'], num_client=args['num_client'])
        gcs = np.zeros((args['ngc'],args['d']))
        for i in range(args['ngc']):
            gcs[i,i] = 1
        
        #print(lcs.shape)
        #gcs/=10
        #print(lcs[0])
        Y1=generate_data(g_cs=gcs,l_cs=lcs[:int(len(lcs)*0.5)],d=args['d'],local_ratio=args['local_ratio'], num_dp=args['num_dp_per_client']//10)
        Y2=generate_data(g_cs=gcs,l_cs=lcs[int(len(lcs)*0.5):],d=args['d'],local_ratio=args['local_ratio'], num_dp=args['num_dp_per_client'])
        #print(Y1.shape)
        #print(Y2.shape)
        #Y = np.concatenate((Y1,Y2), axis=0)
        Y = Y1+Y2
        lpcs = [single_PCA(Yi, args['nlc']+args['ngc']) for Yi in Y]
        singletrainingloss = np.array([single_loss(Y[i], lpcs[i]) for i in range(len(Y))])
        U_glb = initial_u(Y, d=args['d'], ngc=args['nlc']+args['ngc'])
        print("---------------------------------------")
        print('statistical optimal training loss %.4f'% loss(Y, gcs.T, [lc.T for lc in lcs]))
        #print(loss(Y, gcs.T, [lc.T for lc in lcs]))
        #print(U_glb)
        #U = initial_u(Y, d=args['d'], ngc=args['ngc'])
        #print(U)
        Y_test = generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['test_num_dp_per_client'])
        print('statistical optimal test loss: %.4f'% loss(Y_test, gcs.T, [lc.T for lc in lcs]))
        grouploss = lambda gs,ls : np.array([single_loss(Y_test[i], gs[i], ls[i], nov=0) for i in range(len(Y_test))])
        singletestloss = grouploss([gcs.T for ii in range(args['num_client'])], [lc.T for lc in lcs])
        print('statistical optimal individual test loss: %.4f, %.4f'%(np.mean(singletestloss[:len(singletestloss)//2]), np.mean(singletestloss[len(singletestloss)//2:])))


        print("---------------------------------------")
        print('global model test loss: %.4f'%loss(Y_test, U_glb))
        singletestloss = np.array([single_loss(Y_test[i], U_glb) for i in range(len(Y_test))])
        print('global model indiv model test loss: %.4f, %.4f'%(np.mean(singletestloss[:len(singletestloss)//2]), np.mean(singletestloss[len(singletestloss)//2:])))


        print("---------------------------------------")
        print('indiv model training loss: %.4f, %.4f' %(np.mean(singletrainingloss[:len(singletrainingloss)//2]), np.mean(singletrainingloss[len(singletrainingloss)//2:])))

        
        singletestloss = np.array([single_loss(Y_test[i], lpcs[i]) for i in range(len(Y_test))])
        print('indiv model test loss: %.4f, %.4f'%(np.mean(singletestloss[:len(singletestloss)//2]), np.mean(singletestloss[len(singletestloss)//2:])))


        
        print("---------------------------------------")
        #args['nlc'] = 0
        U, V, lv = personalized_pca_dgd(Y, args=args)
            #U, V = personalized_pca_admm(Y, args=args)
        print('personalized model test loss: %.4f'%loss(Y_test, U, V))
        singletestloss = grouploss(U, V)
        print('personalized indiv model test loss: %.4f, %.4f'%(np.mean(singletestloss[:len(singletestloss)//2]), np.mean(singletestloss[len(singletestloss)//2:])))



            #print(Y_test.shape)
            #print(U_p.shape)
            #print(V[0].shape)
        # print(loss(Y_test, U, V))
            #print(U[0])
            #print(V[0][:,0])
            #print(V[0][:,1])
            #print(V[0][:,2])
            #print(U[1])
        print('personalized subspace loss: %.4f, %.4f'%(subspace_error_avg(U,gcs.T), subspace_error_avg(V,[lc.T for lc in lcs])))
        #print(subspace_error_avg(U,gcs.T), subspace_error_avg(V,[lc.T for lc in lcs]))
        #v = pca_by_gd(Y[0],5,0.01,100)
        #print(v)

        #evs, newU = LA.eig(Y[0].T @ Y[0])
        #print(newU[:,0:5])
        #projected = pca(np.transpose(Y[0]), 5)
        #print(projected)
        U, V, lv = two_shot_pca(Y, args=args)
        print("---------------------------------------")

        #print('two shot train loss:')     
        #print(loss(Y, U, V))

        print('two shot test loss: %.4f'% loss(Y_test, U, V))     
        #print(loss(Y_test, U, V))
        singletestloss = grouploss(U, V)
        print('two shot indiv model test loss: %.4f, %.4f'%(np.mean(singletestloss[:len(singletestloss)//2]), np.mean(singletestloss[len(singletestloss)//2:])))


        print('two shot subspace loss: %.4f, %.4f'%(subspace_error_avg(U,gcs.T), subspace_error_avg(V,[lc.T for lc in lcs])))
        #print(subspace_error_avg(U,gcs.T), subspace_error_avg(V,[lc.T for lc in lcs]))
        

    def img_test(inputargs):
        args = {
            'method': 'power',
            'd': 100,
            'num_client': 4,
            'nlc': 100,
            'ngc': 10,
            'num_dp_per_client': 100,
            'global_epochs': 100,
            'local_epochs': 1,
            'n_power': 1,
            'eta': 1e-2,
            #'choice1':1,
            #'adaptivestepsize':1,
            'rho': 1e1,
            'lambda': 0,
            'decay': 1 - 0.1,
            'logprogress':1,
            'precise':1,
            'inverse':1,
        }
        for key in inputargs:
            if key not in {'nlc','ngc'}:
                args[key] = inputargs[key]
        '''
        from misc import Tee
        import time
        import sys

        output_dir = 'outputs/video_'
        jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
        output_dir += jour
        os.makedirs(output_dir, exist_ok=True)
        sys.stdout = Tee(os.path.join(output_dir, 'out.txt'))    
        '''
        np.random.seed(args['seed'])
        print(args)
        from imgpro import gen_img_data
        Y = gen_img_data(args)
        print('number of images %d'%len(Y))
        args['num_client'] = len(Y)
        args['d'] = len(Y[0,0])
        args['num_dp_per_client'] = len(Y[0])
        U_glb = initial_u(Y, d=args['d'], ngc=args['nlc'] + args['ngc'])
        print(U_glb.shape)
        reconstruct0 = (U_glb @ U_glb.T @ (Y[0].T)).T
        #plt.imshow(reconstruct0)
        #plt.axis('off')
        #plt.show()
        # print(U_glb)
        # U = initial_u(Y, d=args['d'], ngc=args['ngc'])
        # print(U)
        
        Y_test = copy.deepcopy(Y) # generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['num_dp_per_client'])
        print('global model test loss:')
        print(loss(Y_test, U_glb))
        
        if args['algorithm'] in {'rpca'}:
            print('solving robust pca via admm')
            U, V = robust_pca_admm(Y,args)
            for figidx in range(len(Y)): 
                print('saving image {}'.format(figidx))
                reconstruct0 = U[figidx]#.T
                plt.imshow(reconstruct0,cmap='gray')
                plt.axis('off')
                plt.savefig('processedframes/'+'rpca_bg_'+str(figidx)+'.png', bbox_inches='tight')

                reconstruct1 = V[figidx]#.T
                plt.imshow(reconstruct1,cmap='gray')
                plt.axis('off')
                #plt.show()
                plt.savefig('processedframes/'+'rpca_cat_'+str(figidx)+'.png', bbox_inches='tight')

          
        else:
            U, V, lv = personalized_pca_dgd(Y, args=args)
            #U, V, lv = two_shot_pca(Y, args=args)
            #print('personalized model test loss:')
            # print(Y_test.shape)
            # print(U_p.shape)
            # print(V[0].shape)

            for figidx in range(len(Y)):
                    
                print('saving image {}'.format(figidx))
                Ui, Vi = generalized_retract(U[figidx], V[figidx])
                reconstruct0 = (Vi@Vi.T@Y[figidx].T)#.T
               
                plt.imshow(reconstruct0,cmap='gray')
                plt.axis('off')
                plt.savefig('processedframes/'+'cat_'+str(figidx)+'.png', bbox_inches='tight')

                reconstruct1 = (Ui@Ui.T @ Y[figidx].T)#.T
                
                plt.imshow(reconstruct1,cmap='gray')
                plt.axis('off')
                #plt.show()
                plt.savefig('processedframes/'+'bg_'+str(figidx)+'.png', bbox_inches='tight')

                
                plt.imshow(reconstruct0+reconstruct1,cmap='gray')
                plt.axis('off')
                #plt.show()
                plt.savefig('processedframes/'+'full_'+str(figidx)+'.png', bbox_inches='tight')


    def debate_test(inputargs):
        args = {
            'method': 'power',
            'd': 100,
            'num_client': 4,
            'nlc': 2,
            'ngc': 2,
            'num_dp_per_client': 100,
            'global_epochs': 20,
            'local_epochs': 1,
            'n_power': 1,
            'eta': 1e0,
            'rho': 100,
            'lambda': 0,
            'decay': 1 - 0.1,
            'logprogress':1,
            #'precise':1,
        }
        '''
        from misc import Tee
        import time
        import sys

        output_dir = 'outputs/debate_'
        jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
        output_dir += jour
        os.makedirs(output_dir, exist_ok=True)
        sys.stdout = Tee(os.path.join(output_dir, 'out.txt'))  
        '''
        np.random.seed(2021)
        print(args)
        from vectorize import vectorize_words, top_words
        Y, number2word, allyears = vectorize_words()
        print('data loaded')
        print('number of elections %d'%len(Y))
        print('number of dialogues %d'%sum([len(Y[i]) for i in range(len(Y))]))
        args['num_client'] = len(Y)
        args['d'] = len(Y[0][0])
        args['num_dp_per_client'] = len(Y[0])
        U_glb = initial_u(Y, d=args['d'], ngc=args['nlc'] + args['ngc'])
        print(U_glb.shape)
        reconstruct0 = (U_glb @ U_glb.T @ (Y[0].T)).T
    
        
        Y_test = copy.deepcopy(Y) # generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['num_dp_per_client'])
        print('global model test loss:')
        print(loss(Y_test, U_glb))

        U, V, lv = personalized_pca_dgd(Y, args=args)
        #print('personalized model test loss:')
            # print(Y_test.shape)
            # print(U_p.shape)
            # print(V[0].shape)

        for yearidx in range(len(Y)):            
            Ui, Vi = generalized_retract(U[yearidx], V[yearidx])
            print('year %d :'% allyears[yearidx])
            words = []
            for j in range(args['nlc']):
                words += top_words(Vi[:,j], number2word, top=10)
            print(list(set(words)))
        print('common words:')
        
        words = []
        for j in range(args['ngc']):
            words += top_words(Ui[:,j], number2word, top=10)
        print(list(set(words)))

            
    def femnist_test(inputargs):
        args = {
            'method': 'power',
            'd': 100,
            'num_client': 4,
            'nlc': 5,
            'ngc': 50,
            'num_dp_per_client': 100,
            'global_epochs': 50,
            'local_epochs': 50,
            'n_power': 1,
            'eta': 1,
            'rho': 100,
            'lambda': 0,
            'decay': 1 - 0.1,
            'logprogress':1,
            'aggregationinit':1,

        }
        '''
        from misc import Tee
        import time
        import sys

        output_dir = 'outputs/femnist_'
        jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
        output_dir += jour
        os.makedirs(output_dir, exist_ok=True)
        sys.stdout = Tee(os.path.join(output_dir, 'out.txt'))
        '''
        # num_client=20
        np.random.seed(4)
        from mnist import femnist_images, femnist_images_labels
        Y, Y_test, lbtrain, lbtest = femnist_images_labels()
        # print(Y)
        #for y in Y:
        #    print(y.shape)
        args['num_client'] = len(Y)
        args['d'] = len(Y[0][0])
        args['num_dp_per_client'] = len(Y[0])
        U_glb = initial_u(Y, d=args['d'], ngc=args['nlc'] + args['ngc'])
        print(U_glb.shape)
        import  imgpro 
        #imgpro.femnist_save_top_eigen(output_dir+'/global_',U_glb,10)
        
        #plt.imshow(reconstruct0)
        #plt.axis('off')
        #plt.show()
        # print(U_glb)
        # U = initial_u(Y, d=args['d'], ngc=args['ngc'])
        # print(U)
        
        #Y_test = copy.deepcopy(Y) # generate_data(g_cs=gcs, l_cs=lcs, d=args['d'], num_dp=args['num_dp_per_client'])
        print('global model train loss:')
        print(loss(Y, U_glb))
        print('global model test loss:')
        print(loss(Y_test, U_glb))


        # logistic regression
        Yr = [U_glb.T@Yi.T for Yi in Y]
        Yrtest = [U_glb.T@Yi.T for Yi in Y_test]
        '''
        trainacc, testacc = logistic_regression(Yr,lbtrain,Yrtest,lbtest)
        print('global model train acc:')
        print(trainacc)
        print('global model test acc:')
        print(testacc)
        

        U, V, lv = personalized_pca_dgd(Y, args=args)
        print('perpca train loss')
        print(loss(Y,U,V))
        print('perpca test loss')
        print(loss(Y_test,U,V))
        ucb = [np.concatenate((U[i],V[i]),axis=1) for i in range(len(V))]
        Yr = [ucb[i].T@Y[i].T for i in range(len(V))]
        Yrtest = [ucb[i].T@Y_test[i].T for i in range(len(V))]
        trainacc, testacc = logistic_regression(Yr,lbtrain,Yrtest,lbtest)
        print('perpcal train acc:')
        print(trainacc)
        print('perpca test acc:')
        print(testacc)


        U, V, lv = two_shot_pca(Y, args=args)
        print('two shot train loss')
        print(loss(Y,U,V))
        print('two shot test loss')
        print(loss(Y_test,U,V))
        ucb = [np.concatenate((U[i],V[i]),axis=1) for i in range(len(V))]
        Yr = [ucb[i].T@Y[i].T for i in range(len(V))]
        Yrtest = [ucb[i].T@Y_test[i].T for i in range(len(V))]
        trainacc, testacc = logistic_regression(Yr,lbtrain,Yrtest,lbtest)
        print('two shot train acc:')
        print(trainacc)
        print('two shot test acc:')
        print(testacc)
        '''
        lpcs = [single_PCA(Yi, args['nlc']+args['ngc']) for Yi in Y]
        singletestloss = np.array([single_loss(Y_test[i], lpcs[i]) for i in range(len(Y_test))])
        print('indiv PCA model test loss: %.4f'%(np.mean(singletestloss[:len(singletestloss)])))
        singletestloss = np.array([single_loss(Y[i], lpcs[i]) for i in range(len(Y_test))])
        print('indiv PCA model train loss: %.4f'%(np.mean(singletestloss[:len(singletestloss)])))

      
    def intro_example(inputargs):
        args = {
            'method':'power',
            'd':3,
            'num_client':2,
            'nlc':1,
            'ngc':1,
            'num_dp_per_client':100,
            'global_epochs':100,
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


    def toy_example1(inputargs):
        import json
        args = {
            'd': 3,
            'num_client': 2,
            'nlc': 1,
            'ngc': 1,
            'num_dp_per_client': 100,
            'global_epochs': 100,
            'choice1':1,
            #'adaptivestepsize':1,            
            #'n_power': 1,
            'eta': 1e-0,
            'rho': 1,
            'decay': 1 - 0.05,
            'randominit':1,
        }

        # num_client=20
        num_runs = 10
        np.random.seed(2021)
        gcs = np.array([[0, 0, 1]])
        resdict = {}
        for alpha in np.linspace(1,90,100):
            resdict[alpha] = []
            for number in range(num_runs):
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
                #U_glb = initial_u(Y, d=args['d'], ngc=args['nlc'] + args['ngc'])
                #scale = 2.1
                #U_glb *= scale

                U, V, lv = personalized_pca_dgd(Y, args=args)
                resdict[alpha].append(lv)
            resdict[alpha] = np.stack(resdict[alpha])
            #print(resdict[alpha].shape)

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
        dev = []
        for alpha in resdict:
            ar = alpha / 180 * np.pi
            theta = 1-np.cos(ar/2)
            tv.append(theta)
            lv.append(np.mean(np.log(resdict[alpha]), axis=0)[-1]/np.log(10))
            dev.append(np.std(np.log(resdict[alpha]), axis=0)[-1]/np.log(10)/np.sqrt(num_runs))
            #plt.plot(range(len(resdict[alpha][0])), np.log(np.mean(resdict[alpha], axis=0)),label=alpha)
        lv = np.array(lv)
        dev = np.array(dev)

        from plotall import CD
        plt.plot(tv, lv, color='red')

        #plt.scatter(tv, lv, color = CD['ppca'])
        #plt.plot(tv, lv, color=CD['ppca'],linestyle='--', label='Personalized PCA')
        plt.fill_between(tv, lv-1.732*dev, lv+1.732*dev, alpha=0.5)

        plt.xlabel(r'$\theta$',fontsize=20)
        plt.ylabel('log reconstruction error',fontsize=20)
        #plt.title('Log training reconstruction error after {} rounds'.format(args['global_epochs']),fontsize=20)
        #plt.legend(fontsize=20)
        plt.savefig('logerrortotheta.png', bbox_inches='tight')


    def toy_example2(inputargs):
        args = {
            'd': 3,
            'num_client': 2,
            'nlc': 1,
            'ngc': 1,
            'num_dp_per_client': 100,
            'global_epochs': 100,
            'choice1':1,
            #'adaptivestepsize':1,            
            #'n_power': 1,
            'eta': 1e-0,
            'rho': 1,
            'decay': 1 - 0.05,
            'randominit':1,
            'alpha':60,
        }

        # num_client=20
        num_runs = 1
        np.random.seed(2021)
        gcs = np.array([[0, 0, 1]])
        color1 = np.array([0.8, 0., 0.])
        color2 = np.array([0., 0.8, 0.8])
        dcolor = (color2-color1)*0.5
        for alpha in range(10,100,10):
            resdict = []
            for number in range(num_runs):
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

                U, V, lv = personalized_pca_dgd(Y, args=args)
                resdict.append(lv)
            resdict = np.stack(resdict)
       
            lv = np.mean(np.log(resdict), axis=0)/np.log(10)
            rounds = np.arange(len(lv))

            plt.plot(rounds, lv, color=color1+theta*dcolor, label='$\theta$=%.2f'%theta)
        from plotall import CD

        #plt.scatter(tv, lv, color = CD['ppca'])
        #plt.plot(tv, lv, color=CD['ppca'],linestyle='--', label='Personalized PCA')
        #plt.fill_between(tv, lv-1.732*dev, lv+1.732*dev, alpha=0.5)

        plt.xlabel(r'Communication round',fontsize=20)
        plt.ylabel('log reconstruction error',fontsize=20)
        #plt.title('Log training reconstruction error after {} rounds'.format(args['global_epochs']),fontsize=20)
        #plt.legend(fontsize=20)
        plt.savefig('logerrortoround.png', bbox_inches='tight')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='personalized pca')
    parser.add_argument('--dataset', type=str, default="borrowpowertest")
    parser.add_argument('--algorithm', type=str, default="dgd")
    parser.add_argument('--logoutput', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--d', type=int, default=15)
    parser.add_argument('--num_client', type=int, default=100)
    parser.add_argument('--nlc', type=int, default=10)
    parser.add_argument('--ngc', type=int, default=2)
    parser.add_argument('--num_dp_per_client', type=int, default=1000)
    parser.add_argument('--folderprefix', type=str, default='')

    args = parser.parse_args()
    args = vars(args)
    if args['logoutput']:
        from misc import Tee
        import time
        import sys
        output_dir = args['folderprefix']+'outputs/{}_'.format(args['dataset'])
        jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
        output_dir += jour
        os.makedirs(output_dir, exist_ok=True)
        sys.stdout = Tee(os.path.join(output_dir, 'out.txt')) 

    experiment = getattr(Experiment, args['dataset'])
    experiment(args)

    #borrowpowertest()
    #img_test()
    #femnist_test()
    #toy_example1()
    #toytest()
    #debate_test()