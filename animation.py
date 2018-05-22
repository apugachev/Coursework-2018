import matplotlib.pyplot as plt
import numpy as np
import sobol_lib as sl
from scipy.spatial import distance
from scipy.sparse import csgraph
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider

np.random.seed(102)

def DotsCoordinates(n, k, a, method):
    '''
    n - dimension of space
    k - number of points
    a - length of hypercube's edge
    method - method of dots generation: 'Uniform', 'Halton' or 'Sobol'

    Returns matrix in which coordinates of point are saved line-by-line
    '''

    if method == 'Uniform':
        result = np.zeros((k, n))
        for i in range(k):
            result[i] = np.random.uniform(0, a, n)

    elif method == 'Sobol':
        result = sl.i4_sobol_generate(n, k, 10)
        result = result.T * a

    return result

def EuclidMatrix(coord_matrix):
    '''
    coord_matrix - матрица с координатами точек (ее возвращает DotsCoordinates)

    Возвращает матрицу, в ячейке [i,j] которой записано евклидово расстояние между i-й и j-й точками матрицы coord_matrix
    '''

    result = np.zeros((len(coord_matrix), len(coord_matrix)))
    for i in range(len(result)):
        for j in range(len(result)):
            result[i, j] = distance.euclidean(coord_matrix[i], coord_matrix[j])

    return result

def CreateAdjMatrix(euclid_matrix, eps):
    '''
    euclid_matrix - матрица евклидовых расстояний между всеми точками пространства
    eps - значение параметра epsilon

    Возвращает матрицу смежности графа, в котором i-й и j-й узлы соединены ребрами
    если евклидовое расстояние между ними меньше epsilon
    '''

    result = np.zeros((len(euclid_matrix), len(euclid_matrix)))
    connected_dots = list(zip(*np.where(euclid_matrix <= eps)))
    for j in range(len(connected_dots)):
        result[connected_dots[j][0]][connected_dots[j][1]] = 1

    return result

def Compute(n, k, a, eps_min, eps_max, steps, show_graph=False, pdf_export=False):
    '''
    n - размерность пространства
    k - количество точек
    a - длина ребра гиперкуба
    eps_min - минимальное значение eps
    eps_max - максимальное значение eps
    steps - количество итераций от eps_min до eps_max
    show_graph - True, если необходимо построить график зависимости количества кластеров от значения epsilon, False иначе
    pdf_export - True, если необходимо экспортировать график в PDF

    Возвращает значение epsilon, при котором образуется кластер
    '''

    matrix = DotsCoordinates(n, k, a, 'Sobol')
    euclid_matrix = EuclidMatrix(matrix)
    components_num = np.array([])
    flag = False
    epsilons = np.arange(eps_min, eps_max, (eps_max - eps_min) / steps)
    for i in range(steps):
        adj_matrix = CreateAdjMatrix(euclid_matrix, epsilons[i])
        components = csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=False)
        components_num = np.append(components_num, components)
        if (components == 1 and flag == False):
            one_component_x = epsilons[i]
            one_component_y = components
            flag = True
            if flag:
                new_eps = one_component_x

    if not flag:
        new_eps = None

    if show_graph:
        fig = plt.figure(figsize=(15, 8))
        plt.plot(epsilons, components_num)
        plt.title('Зависимость количества кластеров от значения $\epsilon$', size=15)
        plt.ylabel('Количество кластеров', size=13)
        plt.xlabel('Значение $\epsilon$', size=13)
        plt.scatter(one_component_x, one_component_y, c='red', marker='o')
        plt.legend()
        plt.grid()
        plt.show()

    if pdf_export:
        pp = PdfPages('Graph.pdf')
        pp.savefig(fig)
        pp.close()

    return new_eps

k = 10

Dots = DotsCoordinates(2,k,10, 'Sobol')

euclid = EuclidMatrix(Dots)

fig = plt.figure(figsize=(10,8))
#plt.subplots_adjust(left=0.15, bottom=0.25)
plt.ion()
plt.grid()
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.pause(5)
for i in range(len(Dots)):
    scat = plt.scatter(Dots[i][0], Dots[i][1], c='red', s=100)
    plt.show()
    plt.pause(0.2)

epsilons = np.arange(1, 10, 1/2)

#axhre = plt.axes([0.15, 0.15, 0.65, 0.03], axisbg='lightgoldenrodyellow')

for eps in epsilons:
    adj_matrix = CreateAdjMatrix(euclid, eps)
    #slid = Slider(axhre, '$\epsilon$', 1, 10, valinit=eps)
    X = np.where(adj_matrix == 1)[0]
    Y = np.where(adj_matrix == 1)[1]
    for i in range(len(X)):
        plt.pause(0.01)
        scat = plt.plot([Dots[X[i]][0], Dots[Y[i]][0]], [Dots[X[i]][1] ,Dots[Y[i]][1]], c='blue', lw=2)
    if csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=False) == 1:
        break

print('Exit')
plt.ioff()
plt.show()

