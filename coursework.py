import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import csgraph
import scipy.stats.stats as st
from tqdm import tqdm
import math
from matplotlib.backends.backend_pdf import PdfPages

def DotsCoordinates(n, k, a):
    
    '''
    n - dimension of space
    k - number of points
    a - length of hypercube's edge
    
    Returns matrix in which coordinates of point are saved line-by-line
    '''
    
    result = np.zeros((k,n))
    for i in range(k):
        result[i] = np.random.uniform(0, a, n)
        
    return result

def EuclidMatrix(coord_matrix):
    
    '''
    coord_matrix - matrix with point coordinates (is returned by DotsCoordinates function)
    
    Returns matrix where in [i,j] cell is written the Euclidean distance between i-th and j-th points of coord_matrix
    '''
    
    result = np.zeros((len(coord_matrix), len(coord_matrix)))
    for i in range(len(result)):
        for j in range(len(result)):
            result[i,j] = distance.euclidean(coord_matrix[i], coord_matrix[j])
            
    return result

def CreateAdjMatrix(euclid_matrix, eps):
    
    '''
    euclid_matrix - Euclidean distance matrix between all of the points
    eps - value of epsilon parameter
    
    Returns the adjacency matrix in which i-th and j-th points are connected with edge 
    if the Euclidean distance between them is less or equal than the value of epsilon
    '''
    
    result = np.zeros((len(euclid_matrix), len(euclid_matrix)))
    connected_dots = list(zip(*np.where(euclid_matrix <= eps)))
    for j in range(len(connected_dots)):
        result[connected_dots[j][0]][connected_dots[j][1]] = 1
        
    return result

def Exp1(n, k, a, eps_min, eps_max, steps, show_graph=False, pdf_export=False):
    
    '''
    n - dimension of space
    k - number of points
    a - length of hypercube's edge
    eps_min - minimum value of epsilon
    eps_max - maximum value of epsilon
    steps - number of iterations from eps_min to eps_max
    show_graph - True, if the plot of dependence between number of clusters and epsilon is needed to be shown, False otherwise
    pdf_export - True, to export the plot to PDF file
    
    Returns epsilon value at which the cluster is formed
    '''
    
    matrix = DotsCoordinates(n, k, a)
    euclid_matrix = EuclidMatrix(matrix)
    components_num = np.array([])
    
    flag = False
    epsilons = np.arange(eps_min, eps_max, (eps_max - eps_min) / steps)
    
    for eps in epsilons:
        adj_matrix = CreateAdjMatrix(euclid_matrix, eps)
        components = csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=False)
        components_num = np.append(components_num, components)
        if (components == 1 and flag == False):
            one_component_x = eps
            one_component_y = components
            flag = True
            if flag:
                new_eps = one_component_x
            
    
    if not flag:
        new_eps = None
    
    if show_graph:
        fig = plt.figure(figsize=(15,8))
        plt.plot(epsilons, components_num)
        plt.title('Dependence between number of clusters and $\epsilon$ value', size=15)
        plt.ylabel('Number of clusters', size=15)
        plt.xlabel('$\epsilon$ value', size=15)
        plt.scatter(one_component_x, one_component_y, c='red', marker='o')
        plt.legend()
        plt.grid()
        plt.show()

    if pdf_export:
        pp = PdfPages('Graph.pdf')
        pp.savefig(fig)
        pp.close()
    
    return new_eps