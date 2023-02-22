import os
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# -------------------------------------------------------------------------------------------------------------
def trainNeuralNetwork(X_train, y_train, X_test, y_test, num_classes, input_dim):
    file_path = os.path.join('.', 'VectorVulnDetector.pkl')

    # If the model already exist, return It, otherwise create It.
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

        # testing the model
        y_pred = model.predict(X_test)

        # processing the testing result in order to convert the output in a format
        # equal to y_test which has been obtained through OneHotEncoding.

        # finding the index of the max element in every row
        max_indices = np.argmax(y_pred, axis=1)

        # creating an array of zeroes with the same shape of the original ndarray
        result = np.zeros_like(y_pred)

        # put 1 only where the maximum element Is.
        result[np.arange(result.shape[0]), max_indices] = 1

        # valuation of the model
        report = classification_report(y_test, result, digits=4)
        print(report)

        # code to save the model after It has been trained
        # with open("VectorVulnDetector.pkl", "wb") as f:
        #    pickle.dump(model, f)
        # return model


# -------------------------------------------------------------------------------------------------------------
# JULIET contains a mix of different level of abstractions in terms of CWE. It has been observed that
# Some CWE are part of some bigger CWE. It has been hypotized that this might be one of many factors that lead
# the model trained on the origianal labels to perform poorly.

# On the left there are the CWE that can join inside the CWE on the right. This will cause a decrease in the total
# number of labels from 112 to 91.
#                                       535 -> 209
#                                       600 -> 248
#                                       759, 760, 319, 328, 614 -> 327
#                                       789 -> 400
#                                       568, 775, 772, 459, 226 -> 404
#                                       510, 511 -> 506
#                                       609, 764, 765, 832, 833 -> 667
#                                       197 -> 681

vectors = []
labels = []

folder_path = 'JULIET_Code_Vectors'

classesNumber = 0

# Through this nested for loop I populate the list "vectors" with all the vectors and the list "labels"
# with all the labels associated to each vector.
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        vectors.extend(data)
        # Labels creation
        for i in range(len(data)):
            label = file_path.split("/")[-1]
            label = label.split(".")[-2]
            # Here I am joining smaller CWEs into bigger CWEs by renaming the label to the Bigger CWE.
            if "535" in label:
                labels.append("CWE209_Information_Leak_Error")
            elif "600" in label:
                labels.append("CWE248_Uncaught_Exception")
            elif "759" in label or "760" in label or "319" in label or "328" in label or "614" in label:
                labels.append("CWE327_Use_Broken_Crypto")
            elif "789" in label:
                labels.append("CWE400_Resource_Exhaustion")
            elif "568" in label or "775" in label or "772" in label or "459" in label or "226" in label:
                labels.append("CWE404_Improper_Resource_Shutdown")
            elif "510" in label or "511" in label:
                labels.append("CWE506_Embedded_Malicious_Code")
            elif "609" in label or "764" in label or "765" in label or "832" in label or "833" in label:
                labels.append("CWE667_Improper_Locking")
            elif "197" in label:
                labels.append("CWE681_Incorrect_Conversion_Between_Numeric_Types")
            else:
                labels.append(label)

print("N. of unique labels {}".format(len(np.unique(labels))))

# -------------------------------Training the Neural Network---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42, stratify=labels)

# Applying OneHotEncoding to y_train and y_test
encoder = LabelBinarizer()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

model = trainNeuralNetwork(X_train, y_train_encoded, X_test, y_test_encoded, 91, 384)

# Results
#               precision    recall  f1-score   support
#
#            0     0.0000    0.0000    0.0000         1
#            1     0.7588    0.9338    0.8372       650
#            2     1.0000    0.8333    0.9091         6
#            3     0.9520    0.6713    0.7874      1299
#            4     1.0000    0.6000    0.7500       325
#            5     0.9916    0.5540    0.7108       213
#            6     0.6818    0.6572    0.6693      2074
#            7     0.5340    0.7963    0.6392      1659
#            8     1.0000    0.9375    0.9677        16
#            9     0.8750    0.8750    0.8750        16
#           10     0.6058    0.6854    0.6432       213
#           11     0.0000    0.0000    0.0000         2
#           12     1.0000    0.8333    0.9091         6
#           13     1.0000    0.8333    0.9091         6
#           14     1.0000    0.8889    0.9412        18
#           15     0.9487    0.6852    0.7957        54
#           16     1.0000    0.7778    0.8750        18
#           17     0.7222    0.7222    0.7222        18
#           18     1.0000    0.5455    0.7059        11
#           19     0.9735    0.9821    0.9778       224
#           20     1.0000    1.0000    1.0000         6
#           21     1.0000    0.6667    0.8000         6
#           22     0.9167    1.0000    0.9565        11
#           23     0.9750    0.7783    0.8656       902
#           24     0.7816    0.6385    0.7028       213
#           25     1.0000    0.8333    0.9091         6
#           26     1.0000    0.6667    0.8000         6
#           27     0.9091    0.9091    0.9091        11
#           28     1.0000    1.0000    1.0000         5
#           29     0.9000    0.8182    0.8571        11
#           30     1.0000    1.0000    1.0000         6
#           31     1.0000    1.0000    1.0000        11
#           32     0.0000    0.0000    0.0000         2
#           33     0.9412    0.7619    0.8421        42
#           34     0.9459    0.8992    0.9220      1459
#           35     0.4681    0.9565    0.6286        23
#           36     0.9147    0.9061    0.9104       213
#           37     0.9620    0.8736    0.9157        87
#           38     0.9500    0.9048    0.9268        21
#           39     1.0000    0.5000    0.6667         6
#           40     0.8333    0.8333    0.8333         6
#           41     1.0000    0.6667    0.8000         6
#           42     1.0000    0.8333    0.9091         6
#           43     1.0000    0.8333    0.9091         6
#           44     1.0000    0.8333    0.9091         6
#           45     0.0000    0.0000    0.0000         1
#           46     1.0000    0.5000    0.6667         2
#           47     0.0000    0.0000    0.0000         1
#           48     0.9459    0.9722    0.9589        72
#           49     1.0000    0.8333    0.9091         6
#           50     0.2564    0.9091    0.4000        11
#           51     0.7500    0.5000    0.6000         6
#           52     0.4000    0.4000    0.4000         5
#           53     1.0000    1.0000    1.0000         6
#           54     0.7143    0.1923    0.3030        26
#           55     0.0000    0.0000    0.0000         6
#           56     1.0000    1.0000    1.0000         1
#           57     1.0000    0.9178    0.9571        73
#           58     1.0000    0.4444    0.6154        18
#           59     0.0000    0.0000    0.0000         5
#           60     0.4286    0.6000    0.5000         5
#           61     1.0000    1.0000    1.0000         6
#           62     0.0000    0.0000    0.0000         1
#           63     0.0000    0.0000    0.0000         1
#           64     0.0000    0.0000    0.0000         2
#           65     0.0000    0.0000    0.0000         1
#           66     0.8333    1.0000    0.9091         5
#           67     0.0000    0.0000    0.0000         1
#           68     1.0000    1.0000    1.0000         6
#           69     1.0000    1.0000    1.0000         6
#           70     0.5000    1.0000    0.6667         6
#           71     0.9196    0.6438    0.7574       160
#           72     1.0000    1.0000    1.0000         6
#           73     0.9948    0.8796    0.9337       216
#           74     0.0000    0.0000    0.0000         1
#           75     1.0000    1.0000    1.0000         6
#           76     1.0000    0.8333    0.9091         6
#           77     0.9000    0.8182    0.8571        11
#           78     0.9781    0.6204    0.7592       216
#           79     0.2632    0.6250    0.3704         8
#           80     0.0000    0.0000    0.0000         1
#           81     0.8816    0.7791    0.8272       602
#           82     0.9925    0.9103    0.9496       145
#           83     1.0000    0.5000    0.6667         6
#           84     0.7450    0.6995    0.7215       213
#           85     0.8601    0.6531    0.7425       320
#           86     0.6042    0.7250    0.6591       160
#           87     0.2500    0.5000    0.3333         2
#           88     0.5407    0.7063    0.6125       160
#           89     0.7375    0.9529    0.8315      1082
#           90     0.8580    0.7089    0.7763       213
#
#    micro avg     0.7729    0.7729    0.7729     13719
#    macro avg     0.7351    0.6676    0.6834     13719
# weighted avg     0.8069    0.7729    0.7769     13719
#  samples avg     0.7729    0.7729    0.7729     13719
