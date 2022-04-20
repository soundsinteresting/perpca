import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from PIL import Image

def read_dir(data_dir, maxuser=10000):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    tot = 0
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        tot += len(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])
        if tot > maxuser:
            break

    clients = list(sorted(data.keys()))
    return clients, groups, data

class FEMNIST:
    N_WORKERS = 0
    def __init__(self, root, hparams,nclient=100):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        #super(ColoredMNIST, self).__init__(root, coloring,
        #                                 self.color_dataset, (2, 28, 28,), 2)
        

        self.datasets = []
        src_dir = os.path.join(root, "leaf/data/femnist/data/all_data")
        clients_raw, groups_raw, data_raw = read_dir(src_dir, nclient)
        print("data loaded")
        for clid, username in enumerate(data_raw.keys()):
            #for order, x in enumerate(data_raw[username]["x"]):
            images = data_raw[username]["x"]
            if len(images) == 0:
                continue
            images = np.array(images)
           
            #images = [np.reshape(img,(1, 28*28))for img in images]
            #images = np.stack(images, axis=0)
            labels = data_raw[username]["y"]
            #print(images.shape)
            #self.datasets.append(TensorDataset(torch.cat((*[tsfm(img) for img in images]), 0), labels))
            self.datasets.append((images,labels))
            #    print(images[0].shape)
        hparams["num_client"] = len(data_raw.keys())
        print("data transformed")        
        
def femnist_images():
    root = "../../../data"
    hparams={"name":"femnist"}
    femnist = FEMNIST(root,hparams)
    images = [image[0] for image in femnist.datasets]
    return images

def femnist_images_labels():
    root = "../../../data"
    hparams={"name":"femnist"}
    femnist = FEMNIST(root,hparams)
    trainlabel = []
    testlabel = []
    trainimage = []
    testimage = []
    testratio = 0.1
    for dataset in femnist.datasets:
        images = dataset[0]
        labels = np.array(dataset[1])
        dsize = len(labels)
        allindices = np.arange(0, dsize)
        #print(labels.shape)
        np.random.shuffle(allindices)
        testsize = int(dsize*testratio)
        testindices = allindices[:testsize]
        trainindices = allindices[testsize:]
        trainlabel.append(labels[trainindices])
        trainimage.append(images[trainindices])
        testlabel.append(labels[testindices])
        testimage.append(images[testindices])
    return trainimage, testimage, trainlabel, testlabel

if __name__ == "__main__":
    data = pd.read_csv('mnist/mnisttrain.csv')
    print(data.head(5))
    l = data['label']
    d = data.drop('label',axis=1 )
    plt.figure(figsize=(7,7))
    idx = 100
    g_data = d.iloc[idx].to_numpy().reshape(28,28)
    plt.imshow(g_data, interpolation='none', cmap='gray')
    plt.savefig('mnist/samplepicture.png')  
    lab = l.head(15000)
    dat = d.head(15000)
    print(d.shape)
    std_data = dat.to_numpy()

    #from sklearn.preprocessing import StandardScaler
    #std_data = StandardScaler().fit_transform(dat)
    print(std_data.shape)

    import seaborn as sn
    from sklearn import decomposition
    pca = decomposition.PCA()

    pca.n_components = 2
    pca_data = pca.fit_transform(std_data)

    pca_data = np.vstack((pca_data.T,lab)).T
    df = pd.DataFrame(data= pca_data, columns= ('1st principle','2nd principle', 'label'))
    sn.FacetGrid(df,hue='label', size=6).map(plt.scatter,'1st principle','2nd principle').add_legend
    plt.savefig('mnist/scoreplot.png')
    plt.close('all')

    print('Calculating svd')
    u,s,vh = np.linalg.svd(std_data.T)
    err = []
    for z in range(200):
        r = z+1
        utop = u[:,:r]
        err.append(1-(np.linalg.norm(utop@utop.T@std_data.T)/np.linalg.norm(std_data))**2)
    err=np.array(err)
    plt.plot(np.linspace(1,200,200),np.log(err))
    plt.savefig('mnist/logresidual.png')
