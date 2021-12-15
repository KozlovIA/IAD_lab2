import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


def point_1_PCA(data, label_list, target):
    plt.figure()
    data_r_2 = PCA(n_components=2, random_state=0)
    data_reduced_2 = data_r_2.fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1], c=target)
    fig = plt.figure()
    ax = Axes3D(fig)
    data_r_3 = PCA(n_components=3, random_state=0)
    data_reduced_3 = data_r_3.fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], s=40, c=target)
    ax.set_title('Visualization')
    ax.set_xlabel(label_list[0])
    ax.set_ylabel(label_list[1])
    ax.set_zlabel(label_list[2])
    print("Выделенные компоненты", data_r_2.components_, "Главные компоненты", data_r_2.explained_variance_ratio_, sep='\n')
    print("Выделенные компоненты", data_r_3.components_, "Главные компоненты", data_r_3.explained_variance_ratio_, sep='\n')
    plt.show()


    
def point_1_TSNE(data, label_list, target, pca=False):
    plt.figure()
    if pca:
        data_reduced_2 = TSNE(n_components=2, init='pca').fit_transform(data)
    else:
        data_reduced_2 = TSNE(n_components=2).fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1], c=target)
    fig = plt.figure()
    ax = Axes3D(fig)
    data_reduced_3 = TSNE(n_components=3).fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], s=30, c=target)
    ax.set_title('Visualization')
    ax.set_xlabel(label_list[0])
    ax.set_ylabel(label_list[1])
    ax.set_zlabel(label_list[2])
    plt.show()

def point_2(data, target):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    Y = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)
    print(Y)
    print("classification_report", classification_report(Y_test, Y), sep='\n')
    print("confusion_matrix", confusion_matrix(Y_test, Y), sep='\n')
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(X_test)
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y_test)
    plt.title('x test, y test')
    plt.show()
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y)
    plt.title('x test, y predict Логистическая регрессия')
    plt.show()
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    Y = model.predict(X_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(X_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y)
    plt.title('x test, y predict метод K-соседей')
    plt.show()
    print("classification_report", classification_report(Y_test, Y), sep='\n')
    print("confusion_matrix", confusion_matrix(Y_test, Y), sep='\n')


def point_3(data, target):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3)
    model = GridSearchCV(KNeighborsClassifier(),
        {'n_neighbors': list(range(2, 6)), 'weights': ['uniform', 'distance']}, n_jobs=3, cv=5)
    print("Обучение модели")
    model.fit(X_train, Y_train)
    print('Best_params')
    print(model.best_params_)
    print('Report')
    print(classification_report(Y_test, model.predict(X_test)))
    model2 = GridSearchCV(LogisticRegression(),
                        {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                         'penalty': ['l1', 'l2'],
                         'C': [i / 10 for i in range(1,15)]},
                         n_jobs=3, cv=5)
    print('Обучение модели')
    model2.fit(X_train, Y_train)
    print('Best params')
    print(model2.best_params_)
    print('Report')
    print(classification_report(Y_test, model.predict(X_test)))


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
    """ print('Обучение модели')
    model2.fit(x_train, y_train)
    print('Best params')
    print(model2.best_params_)
    print('Report')
    print(classification_report(y_test, model.predict(x_test))) """


def point_4(data, lbls, n_neighbors, weights, C, penalty, solver):
    x_train, x_test, y_train, y_test = train_test_split(data, lbls, test_size=0.3)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(x_train, y_train)
    y = model.predict(x_test)
    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(x_test)
    plt.figure()
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=y)
    plt.title('x test, y predict K-ближайших соседей')
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
    plt.title('x test, y predict, Логистическая регрессия')
    plt.show()
    print(classification_report(y_test, y))
    print(confusion_matrix(y_test, y))


def main():
    #Ирисы
    data = sns.load_dataset('iris')
    data_NoLabel = data.drop('species', 1)    #data_NoLabel не содержит столбца с лейблами
    label_list = list(data.columns); label_list.pop(1)
    classes = np.array(data['species'])

    #рост-возраст-вес-позиция
    """ data = pd.read_excel('baseball.xlsx')
    data_NoLabel = data.drop('Position', 1)     #data_NoLabel не содержит столбца с лейблами
    label_list = list(data_NoLabel.columns)
    classes = np.array(data['Position']) """

    target = np.array([0]*len(classes), np.int32)

    k=0
    for i in range(len(classes)-1):
        if classes[i] != classes[i+1]:
            k += 1
            target[i] = k-1
        else:
            target[i] = k
    target[-1] = k


    """ data = pd.read_excel('baseball.xlsx')
    data.sort_values(by='Position', inplace=True)
    data_NoLabel = data.drop(columns='Position')
    labels = data['Position']
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
    target = np.array(sorted(lbls)) """
    #test_data = load_iris()
    
    #print(data_NoLabel)
    #point_1_PCA(data_NoLabel, label_list, target)
    #point_1_TSNE(data_NoLabel, label_list, target, True)
    point_2(data_NoLabel, target)
    #point_3(data_NoLabel, target)
    #point_4(data=data_NoLabel, lbls=target, n_neighbors=6, weights='uniform', C=0.2, penalty='l2', solver='saga')  #ирис
    #point_4(data=data_NoLabel, lbls=target, n_neighbors=2, weights='uniform', C=0.1, penalty='l2', solver='newton-cg')  #рост-возраст-вес
    #rvv()

def rvv():
    data2full = pd.read_excel('baseball.xlsx')
    data2full.sort_values(by='Position', inplace=True)
    data2 = data2full.drop(columns='Position')
    labels = data2full['Position']
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

    


if __name__ == '__main__':
    main()