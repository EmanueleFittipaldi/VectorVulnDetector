import os
import numpy as np
import pickle

from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelBinarizer

# -------------------------------------------------------------------------------------------------------------
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

        #with open("VectorVulnDetector.pkl", "wb") as f:
        #    pickle.dump(model, f)
        #return model



#-------------------------------------------------------------------------------------------------------------
vectors = []
labels = []

folder_path = 'JULIET_Code_Vectors'

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


# -------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42)
# Creazione di un'istanza di LabelBinarizer
encoder = LabelBinarizer()


# Codifica delle label di addestramento
y_train_encoded = encoder.fit_transform(y_train)

# Codifica delle label di test
y_test_encoded = encoder.transform(y_test)

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

model = trainNeuralNetwork(X_train, y_train_encoded, X_test, y_test_encoded, 112, 384)
