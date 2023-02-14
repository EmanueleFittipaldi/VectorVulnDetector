import os

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


# -----------------------------------------------------------------------------------------------------------------------
def trainNeuralNetwork(X_train, y_train, X_test, y_test, num_classes, input_dim):
    file_path = os.path.join('.', 'VectorVulnDetector.pkl')

    if os.path.exists(file_path):
        with open("VectorVulnDetector.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    else:
        # create the model
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32)

        metriche = model.evaluate(X_test,y_test, batch_size=32)
        print("Metriche: {}".format(metriche))

        print("y_test: {}".format(y_test))
        y_pred = model.predict(X_test)

        # Trova l'indice dell'elemento massimo in ogni riga
        max_indices = np.argmax(y_pred, axis=1)

        # Crea un array di zeri con la stessa forma dell'ndarray originale
        result = np.zeros_like(y_pred)

        # Imposta a 1 solo l'elemento massimo in ogni riga
        result[np.arange(result.shape[0]), max_indices] = 1
        print("y_pre: {}".format(result))

        f1 = f1_score(y_test, result, average='macro')
        precision = precision_score(y_test, result, average='macro')
        recall = recall_score(y_test, result, average='macro')
        accuracy = accuracy_score(y_test, result)

        print("F1 score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Accuracy: ", accuracy)

        with open("VectorVulnDetector.pkl", "wb") as f:
            pickle.dump(model, f)
        return model


def trainKMeans(n_clusters, data):
    file_path = os.path.join('.', 'VectorVulnClusters.pkl')

    if os.path.exists(file_path):
        with open("VectorsClusters.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        with open("VectorsClusters.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        return kmeans


# -----------------------------------------------------------------------------------------------------------------------
vectors = []
labels = []

folder_path = 'JULIET_Code_Vectors'

classesNumber = 0

# Through this nested for loop I populate the lists vectors with all the vectors and the list labels
# with all the labels associated for each vector.
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        vectors.extend(data)
        # Labels creation
        for i in range(len(data)):
            label = file_path.split("/")[-1]
            label = label.split(".")[-2]
            labels.append(label)
        classesNumber += 1

clusterNumber = classesNumber
print("N. of CWE classes: {}".format(classesNumber))
print("N. of clusters to be formed: {}".format(clusterNumber))

# -------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42)

# Addestro K-Means su X_train
kmeans = trainKMeans(clusterNumber, X_train)

# Ottengo le labels (assegnamento cluster) per i vettori X_train
temp = X_train
X_train = np.matrix(X_train)
y_train = kmeans.predict(temp)

# Ottengo le labels (assegnamento cluster) per i vettori X_test
temp = X_test
X_test = np.matrix(X_test)
y_test = kmeans.predict(temp)

# create an instance of the encoder
encoder = OneHotEncoder()

# fit the encoder on the labels
encoder.fit(y_train.reshape(-1, 1))

# transform the labels
y_train_encoded = encoder.transform(y_train.reshape(-1, 1))

# conversion from scipy.sparse._csr.csr_matrix to a numpy.ndarray
y_train_encoded = csr_matrix(y_train_encoded).toarray()

encoder2 = OneHotEncoder()
encoder2.fit(y_test.reshape(-1,1))
y_test_encoded = encoder2.transform(y_test.reshape(-1, 1))
y_test_encoded = csr_matrix(y_test_encoded).toarray()

model = trainNeuralNetwork(X_train, y_train_encoded,X_test,y_test_encoded, clusterNumber, 384)
