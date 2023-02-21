import os
import numpy as np
import pickle

from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelBinarizer
from scipy.sparse import csr_matrix


# -------------------------------------------------------------------------------------------------------------
def trainNeuralNetwork(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, input_dim):
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

        # train the model with a validation set and early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[early_stopping])

        # obtain F1, Precision, Recall and Accuracy
        y_pred = model.predict(X_test)

        # finding the index of the max element in every row
        max_indices = np.argmax(y_pred, axis=1)

        # creating an array of zeroes with the same shape of the original ndarray
        result = np.zeros_like(y_pred)

        # put 1 only where the maximum element Is.
        result[np.arange(result.shape[0]), max_indices] = 1

        f1 = f1_score(y_test, result, average='weighted')
        precision = precision_score(y_test, result, average='weighted')
        recall = recall_score(y_test, result, average='weighted')
        accuracy = accuracy_score(y_test, result)

        print("F1 score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Accuracy: ", accuracy)

        # with open("VectorVulnDetector.pkl", "wb") as f:
        #    pickle.dump(model, f)
        # return model


def trainKMeans(n_clusters, data):
    file_path = os.path.join('.', 'VectorVulnClusters.pkl')

    if os.path.exists(file_path):
        with open("VectorsClusters.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        #with open("VectorsClusters.pkl", "wb") as f:
        #    pickle.dump(kmeans, f)
        return kmeans


# -------------------------------------------------------------------------------------------------------------
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

clusterNumber = 91
print("N. of CWE classes: {}".format(classesNumber))
print("N. of clusters to be formed: {}".format(clusterNumber))

# -------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("n.samples in X_train: {}".format(len(X_train)))
print("n.samples in X_test: {}".format(len(X_test)))
print("n.samples in X_val: {}".format(len(X_val)))
print("------------------------------------------------")
# training K-Means on X_train
kmeans = trainKMeans(clusterNumber, X_train)

# obtaining the labels (clusters) for the vectors in X_train
temp = X_train
X_train = np.matrix(X_train)
y_train = kmeans.predict(temp)




# obtaining the labels (clusters) for the vectors in X_test
temp = X_test
X_test = np.matrix(X_test)
y_test = kmeans.predict(temp)


# obtaining the labels (clusters) for the vectors in X_val
temp = X_val
X_val = np.matrix(X_val)
y_val = kmeans.predict(temp)

# -------------------------------------------------------conversion y_train labels to OneHotEncoding---------
encoder2 = LabelBinarizer()
y_train_encoded = encoder2.fit_transform(y_train)
# -------------------------------------------------------conversion y_test labels to OneHotEncoding---------
encoder1 = LabelBinarizer()
y_test_encoded = encoder1.fit_transform(y_test)
# -------------------------------------------------------conversion y_val labels to OneHotEncoding---------
encoder = LabelBinarizer()
y_val_encoded = encoder.fit_transform(y_val)

model = trainNeuralNetwork(X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded, clusterNumber, 384)
