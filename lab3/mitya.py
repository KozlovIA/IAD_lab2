import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


def show_3d_plot(data, k1, k2, k3, c, cmap, edgecolor, s):
    x = data.iloc[:, k1]
    y = data.iloc[:, k2]
    z = data.iloc[:, k3]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(
        x,
        y,
        z,
        c=c,
        cmap=cmap,
        edgecolor=edgecolor,
        s=s
    )
    plt.show()

def task1(data, c):
    plt.figure()
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(data)
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=c)
    plt.show()
    print(model_2.components_, "\n", model_2.explained_variance_ratio_, "\n")
    model_3 = PCA(n_components=3, random_state=0)
    x_reduced_3 = model_3.fit_transform(data)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_reduced_3[:, 0], x_reduced_3[:, 1], x_reduced_3[:, 2], c=c)
    plt.show()
    print(model_3.components_, "\n", model_3.explained_variance_ratio_, "\n")

def task1_without_pca(data, c):
    plt.figure()
    model_2 = TSNE(n_components=2, random_state=0)
    x_reduced = model_2.fit_transform(data)
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=c)
    plt.title('Без PCA')
    plt.show()

def task1_tsne_pca(data, c):
    plt.figure()
    model_2 = TSNE(n_components=2, init='pca', random_state=0)
    x_reduced = model_2.fit_transform(data)
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=c)
    plt.title('С PCA')
    plt.show()


def second_task(data, lbls):
    x_train, x_test, y_train, y_test = train_test_split(data, lbls, test_size=0.3)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y = model.predict(x_test)
    print(classification_report(y_test, y))
    print(confusion_matrix(y_test, y))
    plt.figure()
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(x_test)
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y_test)
    plt.title('x test, y test')
    plt.show()
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y)
    plt.title('x test, y predict LogisticRegression')
    plt.show()
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y = model.predict(x_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(x_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y)
    plt.title('x test, y predict KNeighborsClassifier')
    plt.show()
    print(classification_report(y_test, y))
    print(confusion_matrix(y_test, y))


def third_task(data, lbls):
    x_train, x_test, y_train, y_test = train_test_split(data, lbls, test_size=0.3)
    model = GridSearchCV(KNeighborsClassifier(),
                         {'n_neighbors': list(range(2,6)),
                          'weights': ['uniform', 'distance']},
                         n_jobs=3, cv=5)
    print('Обучение модели')
    model.fit(x_train, y_train)
    print('Best params')
    print(model.best_params_)
    print('Report')
    print(classification_report(y_test, model.predict(x_test)))
    model2 = GridSearchCV(LogisticRegression(),
                        {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                         'penalty': ['l1', 'l2'],
                         'C': [i / 10 for i in range(1,15)]},
                         n_jobs=3, cv=5)
    print('Обучение модели')
    model2.fit(x_train, y_train)
    print('Best params')
    print(model2.best_params_)
    print('Report')
    print(classification_report(y_test, model.predict(x_test)))


def fourth_task(data, lbls, n_neighbors, weights, C, penalty, solver):
    x_train, x_test, y_train, y_test = train_test_split(data, lbls, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(x_train, y_train)
    y = model.predict(x_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(x_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y)
    plt.title('x test, y predict KNeighborsClassifier')
    plt.show()
    print(classification_report(y_test, y))
    print(confusion_matrix(y_test, y))
    model2 = LogisticRegression(C=C, penalty=penalty, solver=solver)
    model2.fit(x_train, y_train)
    y = model2.predict(x_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(x_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y)
    plt.title('x test, y predict, LogisticRegression')
    plt.show()
    print(classification_report(y_test, y))
    print(confusion_matrix(y_test, y))


def iris():
    data1full = pd.read_excel('D:\\ИАД\\Irisdat.xlsx')  # загружается таблица типа DataFrame
    data1full.sort_values(by='species', inplace=True)   # сортируем таблицу по признакам
    data1 = data1full.drop(columns='species')
    labels = data1full['species']
    lbls = [0]*len(labels)  # переименовываем названия признаков в 0, 1 и 2 для удобства
    for i in range(0, len(labels)):
        if labels[i] == 'SETOSA':
            lbls[i] = 0
        elif labels[i] == 'VERSICOL':
            lbls[i] = 1
        elif labels[i] == 'VIRGINIC':
            lbls[i] = 2
    lbls = np.array(sorted(lbls))
    # task1(data1, lbls)
    # task1_without_pca(data1, lbls)
    # task1_tsne_pca(data1, lbls)
    # second_task(data1, lbls)
    third_task(data1, lbls)
    # fourth_task(data1, lbls, 3, 'distance', 0.2, 'l2', 'saga')

def rvv():
    data2full = pd.read_excel('D:\\ИАД\\Рост-Вес-Возраст-Позиция.xlsx')
    data2full.sort_values(by='Позиция', inplace=True)
    data2 = data2full.drop(columns='Позиция')
    labels = data2full['Позиция']
    lbls = [0]*len(labels)
    for i in range(0,len(labels)):
        if labels[i] == 'Baseman':
            lbls[i] = 0
        elif labels[i] == 'Catcher':
            lbls[i] = 1
        elif labels[i] == 'Outfielder':
            lbls[i] = 2
        elif labels[i] == 'Pitcher':
            lbls[i] = 3
        elif labels[i] == 'Relief_P':
            lbls[i] = 4
    lbls = np.array(sorted(lbls))
    # task1(data2, lbls)
    # task1_without_pca(data2, lbls)
    # task1_tsne_pca(data2, lbls)
    #second_task(data2, lbls)
    third_task(data2, lbls)
    # fourth_task(data2, lbls, 2, 'distance', 0.1, 'l2', 'newton-cg')

iris()
# rvv()