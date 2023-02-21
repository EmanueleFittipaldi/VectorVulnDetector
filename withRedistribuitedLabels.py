import os
import numpy as np
import pickle

from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelBinarizer

#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize)


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

        print("y_test: {}".format(result))
        print("y_pred: {}".format(y_test))

        f1 = f1_score(y_test, result, average="weighted")

        # Abilità del classificatore nel non labellare come vero un campione che è falso tp / (tp + fp)
        precision = precision_score(y_test, result, average= "weighted")

        # Abilità del classificatore nel trovare tutti i campioni veri tp / (tp + fn)
        recall = recall_score(y_test, result, average="weighted")

        # Quanto il classificatore si discosta dalla verità
        accuracy = accuracy_score(y_test, result)

        print("F1 score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Accuracy: ", accuracy)

        #with open("VectorVulnDetector.pkl", "wb") as f:
        #    pickle.dump(model, f)
        #return model

# -------------------------------------------------------------------------------------------------------------
# 535 -> 209
# 600 -> 248
# 759, 760, 319, 328, 614 -> 327
# 789 -> 400
# 568, 775, 772, 459, 226 -> 404
# 510, 511 -> 506
# 609, 764, 765, 832, 833 -> 667
# 197 -> 681
# 533, 534 da eliminare -> deprecati
# Le classi passeranno da 112 a 90.
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
            if "535" in label:
                label = "CWE209_Information_Leak_Error"
                labels.append(label)
            elif "600" in label:
                label = "CWE248_Uncaught_Exception"
                labels.append(label)
            elif "759" in label or "760" in label or "319" in label or "328" in label or "614" in label:
                label = "CWE327_Use_Broken_Crypto"
                labels.append(label)
            elif "789" in label:
                label = "CWE400_Resource_Exhaustion"
                labels.append(label)
            elif "568" in label or "775" in label or "772" in label or "459" in label or "226" in label:
                label = "CWE404_Improper_Resource_Shutdown"
                labels.append(label)
            elif "510" in label or "511" in label:
                label = "CWE506_Embedded_Malicious_Code"
                labels.append(label)
            elif "609" in label or "764" in label or "765" in label or "832" in label or "833" in label:
                label = "CWE667_Improper_Locking"
                labels.append(label)
            elif "197" in label:
                label = "CWE681_Incorrect_Conversion_Between_Numeric_Types"
                labels.append(label)
            else:
                labels.append(label)

print("N. of unique labels {}".format(len(np.unique(labels))))

# -------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, stratify=labels)
# Creazione di un'istanza di LabelBinarizer
encoder = LabelBinarizer()

# Codifica delle label di addestramento
y_train_encoded = encoder.fit_transform(y_train)

# Codifica delle label di test
y_test_encoded = encoder.transform(y_test)

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

model = trainNeuralNetwork(X_train, y_train_encoded, X_test, y_test_encoded, 91, 384)
