import csv

from test_HT import extract_features
#from deep_fonts.generate_fonts_ import generate_for_f_and_p

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.models import model_from_json
import numpy
import os

import math

def save(model,name):
    json_path = name+'.json'
    h5_path = name+'.h5'
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_path)
    print("Saved model to disk")

def load(name):
    json_path = name+'.json'
    h5_path = name+'.h5'
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")
    return loaded_model

def predict(model,img,output):
    car = extract_features(img)
    x_data = [car['line'],car['thickness']]
    print(x_data)
    x_data_n = numpy.array(x_data)
    print(x_data_n.shape)
    x_data_fff = [x_data_n]
    model.summary()
    classes = model.predict(numpy.array(x_data_fff),verbose=1)

    count = 0
    index = 0
    value = 0

    for v in classes[0]:
        if v > value:
            value = v
            index = count
        count+=1

    print(index)

    f = int(index/10)
    p_tmp = int(index%10)
    p = 0.0
    if p_tmp == 1 :
        p = 0.111111111111
    elif p_tmp == 2:
        p = 0.222222222222
    elif p_tmp == 3:
        p = 0.333333333333
    elif p_tmp == 4:
        p = 0.444444444444
    elif p_tmp == 5:
        p = 0.555555555556
    elif p_tmp == 6:
        p = 0.666666666667
    elif p_tmp == 7:
        p = 0.777777777778
    elif p_tmp == 8:
        p = 0.888888888889
    elif p_tmp == 9:
        p = 1.0

    print(f,p)

def train(data):

    # This returns a tensor
    inputs = Input(shape=(2,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(128, activation='sigmoid')(inputs)
    x = Dense(256, activation='sigmoid')(x)
    x = Dense(1024, activation='sigmoid')(x)
    predictions = Dense(1000, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    x_train = []
    y_train = []

    with open(data) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            car = extract_features('deep_fonts/'+row['path'])

            x_data = [car['line'],car['thickness']]
            y_data = numpy.zeros(1000, dtype=float)

            f = int(row['f'])
            p = float(row['p'])

            f_int = f*10
            p_int = int(math.floor(p)*10)
            if p == 1.0 :
                p_int = 9

            y_data[f_int+p_int] = 1.0

            x_train.append(numpy.array(x_data))
            y_train.append(y_data)

    x_train_n = numpy.array(x_train)
    y_train_n = numpy.array(y_train)

    model.fit(x_train_n, y_train_n, epochs=5, batch_size=100)

    return model
