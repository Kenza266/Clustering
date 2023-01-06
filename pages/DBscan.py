import numpy as np
import pandas as pd
import streamlit as st
from DBscan import DBscan
from utils import report
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

st.set_page_config(layout="wide")
st.title('Clustering with DBscan')

data = pd.read_csv('datset.csv') 
cat = ['OverTime', 'MaritalStatus', 'JobRole', 'Gender', 'EducationField', 'Department', 'BusinessTravel', 'Attrition']
data = data.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
for i in cat:
    data[i] = (data[i].astype('category').cat.codes).apply(np.int64)
X, y = data.drop(['Attrition'], axis=1), data['Attrition']

X, y = np.array(X), list(y)

data_x = data.drop(['Attrition'], axis=1)
X_standardized = np.array((data_x - data_x.mean()) / data_x.std())
X_normalized = np.array(((data_x - data_x.min()) / (data_x.max() - data_x.min())))

similarity = st.sidebar.selectbox(
    'Distance function',
    ('Manhattan', 'Hamming')) 

if similarity == 'Manhattan':
    preprocessing = st.sidebar.selectbox(
        'Preprocessing',
        ('Raw', 'Norm', 'Std')) 
    if preprocessing == 'Raw':
        input = X
        dist_matrix = np.load('Distance_matrices//Distances_Manhattan.npy')
    elif preprocessing == 'Norm':
        input = X_normalized
        dist_matrix = np.load('Distance_matrices//Distances_Manhattan_Norm.npy')
    elif preprocessing == 'Std':
        input = X_standardized
        dist_matrix = np.load('Distance_matrices//Distances_Manhattan_Std.npy')         

elif similarity == 'Hamming':
    input = X
    dist_matrix = np.load('Distance_matrices//Distances_Hamming.npy')

eps = st.sidebar.number_input('Eps', step=0.1, min_value=0.0, max_value=2500.0, value=10.0)
min_samples = st.sidebar.number_input('MinPts', step=1, min_value=1, max_value=30, value=10)

s = not_inf = ~np.isinf(dist_matrix) 
col1, col2 = st.sidebar.columns([5.3, 1])
col1.write('Distance goes from '+str(format(np.min(dist_matrix), '.2f'))+' to '+str(format(np.max(dist_matrix[s]), '.2f')))

if col2.button('Run'):
    dbscan = DBscan(eps=eps, min_samples=min_samples, similarity=similarity.lower()) 
    clusters = dbscan.cluster(input, dist_matrix=dist_matrix) 
    c1, c2 = st.columns([1, 1.5])
    c1.write('Obtained '+str(len(np.unique(clusters)) - 1)+' clusters, with '+str(list(clusters).count(-2))+' noise points')
    c1.write('Score: '+str(format(adjusted_rand_score(y, clusters), '.6f')))
    c1.write('Distribution: '+str(Counter(clusters)))
    
    reduced = PCA(2).fit_transform(X) 

    c1.text(report(y, list(clusters)))
    
    fig, ax = plt.subplots()
    
    groups = pd.DataFrame(reduced, columns=['x', 'y']).assign(category=clusters).groupby('category')
    for name, points in groups:
        plt.scatter(points.x, points.y, s=5, cmap='Spectral')
    plt.title('Distribution')
    c2.pyplot(fig)