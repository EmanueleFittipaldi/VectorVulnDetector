import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense



def cosineSimilarity(v1, v2):
    inner_product = np.inner(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_sim = inner_product / (norm1 * norm2)
    return cos_sim


def applyPCA(n_components, vectors, n_clusters):
    # Applying PCA in order to reduce the number of components from 384 to 2
    pca = PCA(n_components=n_components)
    code_vectors_2d = pca.fit_transform(vectors)

    # Applying KMeans in order to identify the 3 clusters obtainable from the 2D vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(code_vectors_2d)

    # Plotting the 2D vectors in order to get a glimpse on how these clusters looks like
    fig = plt.gcf()
    plt.figure(figsize=(14, 10), dpi=300)
    plt.scatter(code_vectors_2d[:, 0], code_vectors_2d[:, 1], c=kmeans.labels_, s=0.7)
    plt.show()

def trainNeuralNetwork(X_train,y_train):
    # create the model
    model = Sequential()
    model.add(Dense(64, input_dim=384, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(112, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save('trained_neural_netowrk_JULIET.h5')

vettori = []
folder_path = 'JULIET_Code_Vectors'

for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        vettori.extend(data)



