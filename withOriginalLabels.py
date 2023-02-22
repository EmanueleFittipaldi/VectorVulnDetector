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

        # obtain F1, Precision, Recall and Accuracy
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
vectors = []
labels = []

folder_path = 'JULIET_Code_Vectors'

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
            labels.append(label)

# -------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42, stratify=labels)

print(len(np.unique(y_test)))
print(len(np.unique(y_train)))
# Applying OneHotEncoding to y_train and y_test
encoder = LabelBinarizer()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

model = trainNeuralNetwork(X_train, y_train_encoded, X_test, y_test_encoded, 112, 384)

# Results
#               precision    recall  f1-score   support
#
#            0     0.0000    0.0000    0.0000         1
#            1     0.7605    0.9231    0.8339       650
#            2     0.7143    0.8333    0.7692         6
#            3     0.6335    0.8183    0.7141      1299
#            4     0.9505    0.6492    0.7715       325
#            5     0.5911    0.6854    0.6348       213
#            6     0.6936    0.7903    0.7388      2074
#            7     0.8108    0.6251    0.7059      1659
#            8     0.9412    1.0000    0.9697        16
#            9     0.8044    0.8072    0.8058       586
#           10     0.8182    0.8182    0.8182        11
#           11     1.0000    1.0000    1.0000         6
#           12     0.8851    0.6150    0.7258       213
#           13     0.0000    0.0000    0.0000         1
#           14     1.0000    1.0000    1.0000         6
#           15     1.0000    0.8333    0.9091         6
#           16     1.0000    0.8333    0.9091        18
#           17     0.9524    0.7407    0.8333        54
#           18     1.0000    0.6111    0.7586        18
#           19     0.9833    0.9833    0.9833       180
#           20     0.7778    0.7778    0.7778        18
#           21     0.7500    0.5455    0.6316        11
#           22     0.6154    0.7273    0.6667        11
#           23     1.0000    0.9375    0.9677        16
#           24     1.0000    0.6667    0.8000         6
#           25     1.0000    0.6667    0.8000         6
#           26     0.9091    0.9091    0.9091        11
#           27     0.9972    0.7993    0.8874       902
#           28     0.7030    0.6667    0.6843       213
#           29     1.0000    0.8333    0.9091         6
#           30     1.0000    0.6667    0.8000         6
#           31     1.0000    0.9091    0.9524        11
#           32     1.0000    0.8000    0.8889         5
#           33     1.0000    0.9091    0.9524        11
#           34     1.0000    0.6667    0.8000         6
#           35     1.0000    0.9091    0.9524        11
#           36     1.0000    0.5000    0.6667         2
#           37     0.9024    0.8810    0.8916        42
#           38     0.9986    0.9958    0.9972       709
#           39     0.0000    0.0000    0.0000         2
#           40     1.0000    0.9091    0.9524        11
#           41     0.9458    0.9014    0.9231       213
#           42     1.0000    0.7816    0.8774        87
#           43     1.0000    1.0000    1.0000        21
#           44     1.0000    1.0000    1.0000         6
#           45     0.8333    1.0000    0.9091         5
#           46     1.0000    0.6667    0.8000         6
#           47     1.0000    1.0000    1.0000         6
#           48     1.0000    0.6667    0.8000         6
#           49     1.0000    0.8333    0.9091         6
#           50     0.0000    0.0000    0.0000         1
#           51     1.0000    0.5000    0.6667         2
#           52     0.0000    0.0000    0.0000         1
#           53     0.8718    0.9714    0.9189        35
#           54     1.0000    0.9500    0.9744        20
#           55     0.9412    1.0000    0.9697        16
#           56     1.0000    0.5000    0.6667         6
#           57     1.0000    0.2727    0.4286        11
#           58     0.6667    1.0000    0.8000         6
#           59     1.0000    0.6667    0.8000         6
#           60     1.0000    1.0000    1.0000         6
#           61     0.8333    0.8333    0.8333         6
#           62     0.7742    0.9231    0.8421        26
#           63     0.4000    0.8000    0.5333         5
#           64     0.0000    0.0000    0.0000         1
#           65     0.9103    0.9726    0.9404        73
#           66     1.0000    0.6111    0.7586        18
#           67     1.0000    1.0000    1.0000         2
#           68     0.5000    0.2000    0.2857         5
#           69     0.3333    0.4000    0.3636         5
#           70     1.0000    1.0000    1.0000         6
#           71     0.0000    0.0000    0.0000         1
#           72     0.0000    0.0000    0.0000         1
#           73     0.0000    0.0000    0.0000         2
#           74     1.0000    1.0000    1.0000         1
#           75     1.0000    1.0000    1.0000         6
#           76     0.0000    0.0000    0.0000         1
#           77     1.0000    0.6667    0.8000         6
#           78     1.0000    0.8333    0.9091         6
#           79     0.0000    0.0000    0.0000         6
#           80     0.0000    0.0000    0.0000         1
#           81     0.9802    0.6188    0.7586       160
#           82     1.0000    0.6667    0.8000         6
#           83     0.9510    0.8981    0.9238       216
#           84     0.0000    0.0000    0.0000         1
#           85     0.0000    0.0000    0.0000         1
#           86     1.0000    0.8333    0.9091         6
#           87     0.3125    0.8333    0.4545         6
#           88     1.0000    1.0000    1.0000         6
#           89     1.0000    0.7273    0.8421        11
#           90     0.9710    0.6204    0.7571       216
#           91     0.0000    0.0000    0.0000         1
#           92     0.0000    0.0000    0.0000         1
#           93     1.0000    0.9375    0.9677        16
#           94     0.9466    0.8552    0.8986       145
#           95     1.0000    0.6667    0.8000         6
#           96     1.0000    0.6667    0.8000         6
#           97     0.7143    0.8333    0.7692         6
#           98     0.0000    0.0000    0.0000         1
#           99     0.0000    0.0000    0.0000         1
#          100     0.0000    0.0000    0.0000         1
#          101     0.0000    0.0000    0.0000         1
#          102     0.9009    0.8120    0.8541       750
#          103     0.8720    0.6714    0.7586       213
#          104     0.6170    0.8406    0.7116       320
#          105     0.9899    0.6125    0.7568       160
#          106     0.0000    0.0000    0.0000         1
#          107     0.6667    1.0000    0.8000         2
#          108     1.0000    0.5000    0.6667         2
#          109     0.9561    0.6813    0.7956       160
#          110     0.7397    0.9640    0.8371      1082
#          111     0.9792    0.6620    0.7899       213
#
#    micro avg     0.7932    0.7932    0.7932     13719
#    macro avg     0.7303    0.6437    0.6708     13719
# weighted avg     0.8149    0.7932    0.7936     13719
#  samples avg     0.7932    0.7932    0.7932     13719
