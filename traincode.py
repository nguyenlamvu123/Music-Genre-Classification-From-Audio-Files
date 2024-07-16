import tensorflow as tf
import csv, librosa, os, numpy

import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

from coordinate_constant import (
    epochs, sample_rate, path, header, demo, debug, converter, deploy, genres, readdata, extractfeature, argpa
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.000146)


def predat(pre_dataset=True):
    if pre_dataset:
        my_csv = {
            "filename":[], "chroma_stft_mean": [], "chroma_stft_var": [], "rms_mean": [], "rms_var": [], "spectral_centroid_mean": [],
            "spectral_centroid_var": [], "spectral_bandwidth_mean": [], "spectral_bandwidth_var": [], "rolloff_mean": [], "rolloff_var": [],
            "zero_crossing_rate_mean": [], "zero_crossing_rate_var": [], "harmony_mean": [], "harmony_var": [], "perceptr_mean": [],
            "perceptr_var": [], "tempo": [], "mfcc1_mean": [], "mfcc1_var" : [], "mfcc2_mean" : [], "mfcc2_var" : [],
            "mfcc3_mean" : [], "mfcc3_var" : [], "mfcc4_mean" : [], "mfcc4_var" : [], "mfcc5_mean" : [], 
            "mfcc5_var" : [], "mfcc6_mean" : [], "mfcc6_var" : [], "mfcc7_mean" : [], "mfcc7_var" : [],
            "mfcc8_mean" : [], "mfcc8_var" : [], "mfcc9_mean" : [], "mfcc9_var" : [], "mfcc10_mean" : [], 
            "mfcc10_var" : [], "mfcc11_mean" : [], "mfcc11_var" : [], "mfcc12_mean" : [], "mfcc12_var" : [], 
            "mfcc13_mean" : [], "mfcc13_var" : [], "mfcc14_mean" : [], "mfcc14_var" : [], "mfcc15_mean" : [], 
            "mfcc15_var" : [], "mfcc16_mean" : [], "mfcc16_var" : [], "mfcc17_mean" : [], "mfcc17_var" : [], 
            "mfcc18_mean" : [], "mfcc18_var" : [], "mfcc19_mean" : [], "mfcc19_var" : [], "mfcc20_mean" : [], 
            "mfcc20_var":[], "label":[]
        }
        file = open(f'{path}/try_running__features_3_sec.csv', 'w', newline='')
        with file:  
            writer = csv.writer(file)
            writer.writerow(header)
        for g in genres:
            print(g)
            ldir = sorted(list(os.listdir(f'{path}/Data/genres_original/{g}')))
            listfn = ldir[:5] if demo else ldir
            i = 0
            for filename in listfn:
                songname = f'{path}/Data/genres_original/{g}/{filename}'
                try:
                    y, sr = librosa.load(songname, sr=sample_rate)
                except Exception as e:
                    print(e, 'in', songname)
                    continue
                i_ = 0
                t1 = 0
                totalsecond = y.shape[0] // sample_rate
                while t1 < totalsecond:
                    t1 = i_ * 30 * sample_rate
                    t2 = (i_ + 1) * 30
                    if t2 > totalsecond:
                        break
                    t2 *= sample_rate
                    y_seg = y[t1:t2]
                    fseg_name = filename.split('.')[0]+f'.{i_}.wav'
                    try:
                        my_csv = extractfeature(y_seg, sr, fseg_name, g, my_csv)
                        my_csv["label"].append(g)
                    except ValueError:
                        if demo:
                            # ValueError: can't extend empty axis 0 using modes other than 'constant' or 'empty'
                            print('error in ' + fseg_name)
                            sf.write(fseg_name, y_seg, sample_rate)
                    i += 1
                    if i > 499:
                        break
                    i_ += 1
                    t1 /= sample_rate
                if i > 499:
                    break
        df = pd.DataFrame(my_csv)
        if demo:
            amount_of_each_genre(my_csv)
        df.to_csv(f"{path}/try_running__features_3_sec.csv", index=False)
    return pd.read_csv(f"{path}/try_running__features_3_sec.csv")


def amount_of_each_genre(my_csv):
    assert 'label' in my_csv
    lab = tuple(l for l in my_csv['label'])
    for lab_set in set(lab):
        print(f'{lab.count(lab_set)} samples are found in {lab_set}')


def train_model(model, epochs, optimizer):
    batch_size = 256
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)


def Validation_plot(history):
    print("Validation Accuracy", max(history.history["val_accuracy"]))
    hist_df = pd.DataFrame(history.history)
    hist_df.plot(figsize=(12, 6))
    with open('traininghistory.json', mode='w') as f:
        hist_df.to_json(f)
    plt.savefig('Validation_plot.jpg')


def big_cnn_model(train=True):
    # https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
    # https://medium.com/swlh/saving-and-loading-of-keras-sequential-and-functional-models-b0092fff335
    if not train:
        model = tf.keras.models.load_model("my_model.h5")
##        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        # We used different layers to train the neural network by importing keras library from tensorflow framework
        # for input and hidden neurons we use the most widly used activation function which is relu where as for output neurons we uses softmax activation function
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(10, activation='softmax'),
        ])

        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()
        model_history = train_model(model=model, epochs=epochs, optimizer='adam')

        test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=256)
        print("The test loss is ", test_loss)
        print("The best accuracy is: ", test_acc * 100)

        try:
            Validation_plot(model_history)
        except ValueError:
            print('ValueError: PyCapsule_New called with null pointer')
        finally:
            model.save("my_model.h5")
    return model


def cnn_model(train=True):
    # https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
    if not train:
        model = tf.keras.models.load_model("model.keras")
    else:
        # We used different layers to train the neural network by importing keras library from tensorflow framework
        # for input and hidden neurons we use the most widly used activation function which is relu where as for output neurons we uses softmax activation function
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.000146)
        model.compile(optimizer=optimizer,
                     loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()
        model_history=train_model(model=model, epochs=epochs, optimizer='adam')

        test_loss,test_acc=model.evaluate(X_test,y_test,batch_size=256)
        print("The test loss is ",test_loss)
        print("The best accuracy is: ",test_acc*100)

        Validation_plot(model_history)
        model.save("model.keras")
    return model


if __name__ == '__main__':
    import json
    def pred_a_song(
            # songname = r'\\tsclient\_home_zaibachkhoa_Music\ms_nt_3414_4003.wav',  # 90
            # songname = "/home/bhm-ai/Music/ms_nt_3414_4003.wav",  # 91
            songname = "/home/zaibachkhoa/Music/ms_nt_3414_4003.wav",  # local
    ) -> str :
        y, sr = librosa.load(songname, sr=sample_rate)
        y_seg = y[y.shape[0] // 2 - sample_rate * 50 : y.shape[0] // 2 + sample_rate * 50]
        my_3_csv = {
            "filename":[], "chroma_stft_mean": [], "chroma_stft_var": [], "rms_mean": [], "rms_var": [], "spectral_centroid_mean": [],
            "spectral_centroid_var": [], "spectral_bandwidth_mean": [], "spectral_bandwidth_var": [], "rolloff_mean": [], "rolloff_var": [],
            "zero_crossing_rate_mean": [], "zero_crossing_rate_var": [], "harmony_mean": [], "harmony_var": [], "perceptr_mean": [],
            "perceptr_var": [], "tempo": [], "mfcc1_mean": [], "mfcc1_var" : [], "mfcc2_mean" : [], "mfcc2_var" : [],
            "mfcc3_mean" : [], "mfcc3_var" : [], "mfcc4_mean" : [], "mfcc4_var" : [], "mfcc5_mean" : [],
            "mfcc5_var" : [], "mfcc6_mean" : [], "mfcc6_var" : [], "mfcc7_mean" : [], "mfcc7_var" : [],
            "mfcc8_mean" : [], "mfcc8_var" : [], "mfcc9_mean" : [], "mfcc9_var" : [], "mfcc10_mean" : [],
            "mfcc10_var" : [], "mfcc11_mean" : [], "mfcc11_var" : [], "mfcc12_mean" : [], "mfcc12_var" : [],
            "mfcc13_mean" : [], "mfcc13_var" : [], "mfcc14_mean" : [], "mfcc14_var" : [], "mfcc15_mean" : [],
            "mfcc15_var" : [], "mfcc16_mean" : [], "mfcc16_var" : [], "mfcc17_mean" : [], "mfcc17_var" : [],
            "mfcc18_mean" : [], "mfcc18_var" : [], "mfcc19_mean" : [], "mfcc19_var" : [], "mfcc20_mean" : [],
            "mfcc20_var":[], "label":['_']
        }
        my_csv = extractfeature(y_seg, sr, my_3_csv=my_3_csv)
        df = pd.DataFrame(my_csv)
        # df.to_csv("try_running__Extracted___Data.csv", index=False)
        # df = pd.read_csv('try_running__Extracted___Data.csv')
        X, df = readdata(df)
        pred_x = model.predict(X)
        pred_ind = numpy.argmax(pred_x, axis=1)
        return converter.inverse_transform(pred_ind)[:1][0]


    pre_dataset = not deploy
    df = predat(pre_dataset)

    class_encod = df.iloc[:, -1]
    if debug:
        print('%$%$%$%$%$%$%$%$%$%$%$%$%', class_encod)
    Y = converter.fit_transform(class_encod)
    X, df = readdata(df)
    assert len(list(df.columns[df.isnull().any()])) == 0
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    train = not deploy
    # model = big_cnn_model(train)
    model = cnn_model(train)

    ret: dict = {}
    if deploy:
        args = argpa()
        if os.path.isfile(args.duongdan):
            label = pred_a_song(args.duongdan)
            ret[label].append(args.duongdan)
        elif os.path.isdir(args.duongdan):
            for filename in os.listdir(args.duongdan):
                label = pred_a_song(os.path.join(args.duongdan, filename))
                if label not in ret:
                    ret[label] = [filename, ]
                else:
                    ret[label].append(filename)
        print(json.dumps(ret, sort_keys=True, indent=4))
    else:
        label = pred_a_song()
        print(label)
