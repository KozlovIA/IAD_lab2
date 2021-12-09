import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import cluster
from scipy.sparse.construct import random
import seaborn as sns
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def readFile(file = "data.txt"):  # считывание из файла
    """Reading from txt file
    file - file name"""
    x = []; xNoStr = []    #Все данные и данные чисто числа без строк
    file = open(file)
    for line in file:
        values = list(map(str, line.split()))
        temp = []; tempNoStr = []
        for val in values:
            try:
                temp.append(float(val))
                tempNoStr.append(float(val))
            except:
                temp.append(val)
        x.append(temp); xNoStr.append(tempNoStr)
    file.close()
    return x, xNoStr

def point_1(data):
    """Первый пункт"""
    plt.figure()
    sns.heatmap(data.corr(), annot=True)
    plt.show()

def point_1_1(data, label_list, labels=None):
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    ax.scatter(data[label_list[0]], data[label_list[1]], data[label_list[2]], cmap=plt.cm.Set1, edgecolor='k', c=labels)
    ax.set_title('Visualization')
    ax.set_xlabel(label_list[0])
    ax.set_ylabel(label_list[1])
    ax.set_zlabel(label_list[2])
    plt.show()
    pass

def point_2(data):
    methods = ['single', 'complete', 'average']
    plot_dendrogram(data, distance='euclidean', method='centroid', labels=[])
    plt.title('Метод = centroid' + '\nРасстояние = euclidean')
    plot_dendrogram(data, distance='euclidean', method='ward', labels=[])
    plt.title('Метод = ward' + '\nРасстояние = euclidean')
    for method in methods:
        for dist in ['euclidean', 'chebyshev', 'minkowski', 'cosine']:
            plot_dendrogram(data, distance=dist, method=method, labels=[])
            plt.title(f'Метод = {method}' + f'\nРасстояние = {dist}')
    plt.show()

def plot_dendrogram(data, distance, method, labels):
    linkMatrix = linkage(data, metric=distance, method=method)
    plt.figure(figsize=(15, 8))
    dendrogram(linkMatrix)

def point_3(data, label_list):
    plt.figure()
    data_r_2 = PCA(n_components=2, random_state=0)
    data_reduced_2 = data_r_2.fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1])
    fig = plt.figure()
    ax = Axes3D(fig)
    data_r_3 = PCA(n_components=3, random_state=0)
    data_reduced_3 = data_r_3.fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], s=40)
    ax.set_title('Visualization')
    ax.set_xlabel(label_list[0])
    ax.set_ylabel(label_list[1])
    ax.set_zlabel(label_list[2])
    print("Выделенные компоненты", data_r_2.components_, "Главные компоненты", data_r_2.explained_variance_ratio_, sep='\n')
    print("Выделенные компоненты", data_r_3.components_, "Главные компоненты", data_r_3.explained_variance_ratio_, sep='\n')
    plt.show()


    
def point_4(data, label_list):
    plt.figure()
    data_reduced_2 = TSNE(n_components=2, init='pca').fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1])
    fig = plt.figure()
    ax = Axes3D(fig)
    data_reduced_3 = TSNE(n_components=3).fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], s=30)
    ax.set_title('Visualization')
    ax.set_xlabel(label_list[0])
    ax.set_ylabel(label_list[1])
    ax.set_zlabel(label_list[2])
    plt.show()

def point_6(data, label_list, labels):
    k = 3   #кол-во кластеров
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    print(kmeans.labels_)
    print(labels)
    print("Ошибка", kmeans.inertia_)  #Сумма квадратов расстояний от выборок до ближайшего центра кластера

    #силуэт
    sil_val = silhouette_samples(data, kmeans.labels_)
    plot_silhouette(sil_val, kmeans.labels_)

    #point_7
    point_1_1(data, label_list, kmeans.labels_)


def plot_silhouette(data, labels):
    n_clusters = len(set(labels))
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_val = data[labels == i]
        cluster_silhouette_val.sort()
        y_upper = y_lower + cluster_silhouette_val.shape[0]
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_val)
        plt.text(-0.05, y_lower + 0.5*cluster_silhouette_val.shape[0], str(i+1))
        y_lower = y_upper + 10
        plt.axvline(x=data.mean(), c='r', lineStyle='--')
        plt.title('Silhouette plot')
        plt.xlabel('Silhouette coefficient values')
        plt.ylabel('Cluster labels')
    plt.show()


def main():
    #Ирис
    data = sns.load_dataset('iris')
    data_NoLabel = data.drop('species', 1)    #data_NoLabel не содержит столбца с лейблами
    label_list = list(data.columns); label_list.pop(1)
    labels = np.array(data['species'])

    #рост-возраст-вес-позиция
    """ data = pd.read_excel('baseball.xlsx')
    data_NoLabel = data.drop('Position', 1)     #data_NoLabel не содержит столбца с лейблами
    label_list = list(data_NoLabel.columns)
    labels = np.array(data['Position']) """

    k=0
    for i in range(len(labels)-1):
        if labels[i] != labels[i+1]:
            k += 1
            labels[i] = k-1
        else:
            labels[i] = k
    labels[-1] = k
    #print(data)
    #point_1(data)
    #point_1_1(data, label_list)#, labels)
    #point_2(data_NoLabel)
    #point_3(data_NoLabel, label_list)
    #point_4(data_NoLabel, label_list)
    #point_6(data_NoLabel, label_list, labels)

    #7 принадлежность к реальному кластеру
    point_1_1(data, label_list, labels)



if __name__ == "__main__":
    main()