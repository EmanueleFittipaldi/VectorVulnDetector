import os
import numpy as np
import pickle

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer


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

        # valuation of the model
        report = classification_report(y_test, result, digits=4)
        print(report)

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
        # with open("VectorsClusters.pkl", "wb") as f:
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
# Splitting of the dataset into Training, Test, Validation set.
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42, stratify=labels)
print("unique y_train: {}".format(len(np.unique(y_train))))
print("unique y_test: {}".format(len(np.unique(y_test))))
print("n. X_train: {}".format(len(X_train)))
print("n. X_test: {}".format(len(X_test)))
print("------------------------------------------------------------------------------------------------------")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print("unique y_train: {}".format(len(np.unique(y_train))))
print("unique y_val: {}".format(len(np.unique(y_val))))
print("n. X_train: {}".format(len(X_train)))
print("n. X_val: {}".format(len(X_val)))



print("------------------------------------------------------------------------------------------------------")
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

# Result
#               precision    recall  f1-score   support
#
#            0     1.0000    0.9769    0.9883       130
#            1     0.9699    1.0000    0.9847       129
#            2     0.9762    0.9140    0.9441       314
#            3     0.9243    0.9243    0.9243       251
#            4     1.0000    1.0000    1.0000       389
#            5     0.9792    0.9592    0.9691       196
#            6     1.0000    1.0000    1.0000       262
#            7     1.0000    1.0000    1.0000        28
#            8     1.0000    1.0000    1.0000       145
#            9     0.9441    0.9696    0.9567       296
#           10     1.0000    1.0000    1.0000       154
#           11     0.9783    1.0000    0.9890       721
#           12     1.0000    1.0000    1.0000       186
#           13     0.9864    1.0000    0.9932       145
#           14     0.8943    0.9834    0.9368       482
#           15     1.0000    1.0000    1.0000        65
#           16     0.9942    1.0000    0.9971       345
#           17     1.0000    1.0000    1.0000        86
#           18     1.0000    1.0000    1.0000        41
#           19     0.9921    0.9921    0.9921       127
#           20     1.0000    1.0000    1.0000       131
#           21     1.0000    1.0000    1.0000       538
#           22     1.0000    0.9859    0.9929        71
#           23     0.9867    0.9737    0.9801        76
#           24     0.9857    0.9801    0.9829       702
#           25     0.9611    0.9825    0.9717       629
#           26     1.0000    0.9643    0.9818        28
#           27     1.0000    1.0000    1.0000        63
#           28     1.0000    1.0000    1.0000        77
#           29     0.9688    0.9894    0.9789        94
#           30     0.9891    0.8778    0.9302       311
#           31     1.0000    1.0000    1.0000       117
#           32     1.0000    1.0000    1.0000        23
#           33     1.0000    1.0000    1.0000       135
#           34     1.0000    1.0000    1.0000        14
#           35     0.9935    1.0000    0.9967       152
#           36     1.0000    1.0000    1.0000        56
#           37     1.0000    1.0000    1.0000       112
#           38     1.0000    1.0000    1.0000        47
#           39     1.0000    1.0000    1.0000        39
#           40     1.0000    1.0000    1.0000        50
#           41     1.0000    1.0000    1.0000       166
#           42     1.0000    1.0000    1.0000        70
#           43     1.0000    1.0000    1.0000       177
#           44     1.0000    1.0000    1.0000        52
#           45     0.9804    0.9804    0.9804        51
#           46     0.9673    0.8636    0.9125       308
#           47     1.0000    0.9714    0.9855        35
#           48     1.0000    0.9827    0.9913       173
#           49     1.0000    1.0000    1.0000        81
#           50     1.0000    1.0000    1.0000        51
#           51     0.9975    0.9052    0.9491       443
#           52     0.8837    0.9520    0.9166       375
#           53     1.0000    1.0000    1.0000        45
#           54     1.0000    1.0000    1.0000       110
#           55     1.0000    1.0000    1.0000        57
#           56     1.0000    1.0000    1.0000        37
#           57     1.0000    1.0000    1.0000        53
#           58     0.9718    1.0000    0.9857        69
#           59     0.8683    0.9889    0.9247       180
#           60     1.0000    1.0000    1.0000        23
#           61     0.9740    0.9868    0.9804       152
#           62     0.9886    0.9422    0.9649       277
#           63     1.0000    0.9833    0.9916        60
#           64     0.9779    1.0000    0.9888       177
#           65     0.9787    1.0000    0.9892        46
#           66     1.0000    1.0000    1.0000        15
#           67     1.0000    1.0000    1.0000        18
#           68     1.0000    1.0000    1.0000        16
#           69     0.9947    0.9843    0.9895       191
#           70     0.9426    0.9829    0.9623       234
#           71     0.9559    0.9848    0.9701        66
#           72     0.9836    0.9836    0.9836        61
#           73     1.0000    1.0000    1.0000        78
#           74     1.0000    1.0000    1.0000        49
#           75     1.0000    1.0000    1.0000        34
#           76     1.0000    1.0000    1.0000        18
#           77     1.0000    1.0000    1.0000        57
#           78     1.0000    0.9867    0.9933        75
#           79     1.0000    0.9839    0.9919        62
#           80     1.0000    1.0000    1.0000         9
#           81     1.0000    1.0000    1.0000        47
#           82     0.9877    0.9450    0.9659       509
#           83     0.9867    0.9024    0.9427        82
#           84     1.0000    1.0000    1.0000        59
#           85     1.0000    1.0000    1.0000        27
#           86     1.0000    1.0000    1.0000       196
#           87     0.9930    0.9726    0.9827       146
#           88     1.0000    1.0000    1.0000        24
#           89     0.9657    0.9826    0.9741       172
#           90     0.9077    0.9880    0.9462       249
#
#    micro avg     0.9761    0.9761    0.9761     13719
#    macro avg     0.9871    0.9860    0.9863     13719
# weighted avg     0.9771    0.9761    0.9761     13719
#  samples avg     0.9761    0.9761    0.9761     13719