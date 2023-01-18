import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt


def cosineSimilarity(v1, v2):
    inner_product = np.inner(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_sim = inner_product / (norm1 * norm2)
    return cos_sim


def strListTofloatList(v1):
    v1 = v1.strip('[]')
    v1 = v1.split(' ')
    # deletes empty strings caused by the spaces between the values contained by the code vector
    v1 = [string for string in v1 if string.strip() != '']
    # delete the \n characters from the values containing them
    v1 = [string.replace('\n', '') for string in v1]
    # convert the string list into a float list
    float_list = list(map(float, v1))
    return float_list


vettori = []
i = 0
# This file is too big to commit, I'll upload a link where to download it
path = "JULIET_Code_Vectors/Code_Vectors.csv"
df = pd.read_csv(path)

# Taking all the vectors contained in the csv
for b in range(df.shape[0]):
    vettori.append(strListTofloatList(df.iloc[i]['Code_Vectors']))
    i += 1

# Applying PCA in order to reduce the number of components from 384 to 2
pca = PCA(n_components=2)
code_vectors_2d = pca.fit_transform(vettori)

# Applying KMeans in order to identify the 3 clusters obtainable from the 2D vectors
kmeans = KMeans(n_clusters=3, random_state=0).fit(code_vectors_2d)

# Plotting the 2D vectors in order to get a glimpse on how these clusters looks like
fig = plt.gcf()
plt.figure(figsize=(14,10), dpi=300)
plt.scatter(code_vectors_2d[:, 0], code_vectors_2d[:, 1], c=kmeans.labels_, s=0.01)
plt.show()


