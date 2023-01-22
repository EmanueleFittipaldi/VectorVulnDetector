import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from scipy.sparse import csr_matrix


def cosineSimilarity(v1, v2):
    inner_product = np.inner(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_sim = inner_product / (norm1 * norm2)
    return cos_sim


def applyPCAandKMeans(n_components, vectors, n_clusters):
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


def trainNeuralNetwork(X_train, y_train, num_classes, input_dim):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    # create the model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)


def train_and_evaluate_random_forest(X, y, test_size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create the model
    clf = RandomForestClassifier(n_estimators=100)

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.3f}".format(acc))


# --------------------------------------------------------------------------------------
# This list will contain every vector of every CWE
vettori = []

# This list will contain the CWE label for every vector. We will apply oneHotEncoding
# so that we can train an AI model afterwards.
labels = []
datasetExploration = {}

folder_path = 'JULIET_Code_Vectors'
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        # Printing which CWE needs more samples
        if len(data)<200:
            print("POOR SAMPLES NUMBER")
            print(((file_path.split("/")[-1]).split("."))[-2]+":  " +str(len(data)))
        datasetExploration[((file_path.split("/")[-1]).split("."))[-2]] = len(data)
        # Labels creation
        for i in range(len(data)):
            label = file_path.split("/")[-1]
            label = label.split(".")[-2]
            labels.append(label)
        vettori.extend(data)

# -------------Printing of the dataset state------------
datasetExploration = dict(sorted(datasetExploration.items(), key=lambda x:x[1]))

# Creazione del grafico a barre
plt.figure(figsize=(65, 53))
plt.bar(datasetExploration.keys(), datasetExploration.values())


# Assegnazione del titolo e delle etichette per gli assi
plt.title("JULIET Dataset vectors distribuition",fontsize=50)
plt.ylabel("N. of samples",fontsize=50)
plt.yticks(fontsize=20)
plt.xticks(rotation=90,fontsize=13)

# Visualizzazione del grafico
plt.show()

# --------Shaping vectors list and vectors labels in order to train a neural network

X_train = np.matrix(vettori)
labels = np.array(labels)

# create an instance of the encoder
encoder = OneHotEncoder()

# fit the encoder on the labels
encoder.fit(labels.reshape(-1, 1))

# transform the labels
y_train_encoded = encoder.transform(labels.reshape(-1, 1))

# conversion from scipy.sparse._csr.csr_matrix to a numpy.ndarray
y_train_encoded = csr_matrix(y_train_encoded).toarray()

trainNeuralNetwork(X_train, y_train_encoded, 112, 384)
#train_and_evaluate_random_forest(vettori, y_train_encoded)
