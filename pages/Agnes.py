import numpy as np
import pandas as pd
import streamlit as st
from Agnes import Agnes
from utils import report
import matplotlib.pyplot as plt 

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering as SK_Agnes

st.set_page_config(layout="wide")
st.title('Clustering with Agnes')

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
        file = 'Distance_matrices//Distances_Manhattan.npy'
    elif preprocessing == 'Norm':
        input = X_normalized
        file = 'Distance_matrices//Distances_Manhattan_Norm.npy'
    elif preprocessing == 'Std':
        input = X_normalized
        file = 'Distance_matrices//Distances_Manhattan_Std.npy'        

elif similarity == 'Hamming':  
    input = X
    file = 'Distance_matrices//Distances_Hamming.npy'

link = st.sidebar.selectbox(
    'Linkage',
    ('Average', 'Complete', 'Single')) 

agnes = Agnes(similarity.lower()) 
dist_matrix = np.load(file, 'r') 

if st.sidebar.button('Run'):
    agnes = SK_Agnes(2, affinity=similarity.lower(), linkage=link.lower())
    clustering = agnes.fit(input) 

    col1, col2 = st.columns(2)

    col1.text(report(y, list(clustering.labels_)))
    col1.text('Silhouette score: ' + str(silhouette_score(input, list(clustering.labels_))))
    col1.text('Adjusted rand score: ' + str(adjusted_rand_score(y, list(clustering.labels_))))

    linked = linkage(input, link.lower()) 
    labelList = range(1, len(input) + 1)

    fig, ax = plt.subplots()
    dendrogram(linked,
                orientation='right',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True,above_threshold_color='#b329dd')
    plt.axis(False)
    col2.pyplot(fig)