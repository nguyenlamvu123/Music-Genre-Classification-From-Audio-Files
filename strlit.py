from coordinate_constant import extractfeature, readdata, converter, header, argpa, sample_rate, timer, debug
from traincode import big_cnn_model, predat, optimizer
import librosa, csv, os
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def predict2label(res):
    return converter.inverse_transform(res)

@timer
def main(lis=None):
    args = argpa() if lis is None else argpa([lis])
    songname = args.duongdan
    print(songname)

    df = predat(False)
    class_encod = df.iloc[:, -1]
    if debug:
        print('%$%$%$%$%$%$%$%$%$%$%$%$%', class_encod)
    Y = converter.fit_transform(class_encod)
    X, df = readdata(df)
    assert len(list(df.columns[df.isnull().any()])) == 0
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

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

    model = big_cnn_model(False)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('**************************************************')
    loss, acc_tf = model.evaluate(X_test, y_test, verbose=1)
    print('**************************************************')
    ##print("Loaded model (h5), accuracy: {:5.2f}%".format(100*acc_tf))

    if debug:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train,y_train)

        dt = DecisionTreeClassifier(criterion="entropy")
        dt.fit(X_train,y_train)

        rf= RandomForestClassifier(n_estimators=100,random_state=0)
        rf.fit(X_train,y_train)

    y, sr = librosa.load(songname, sr=sample_rate)
    y_seg = y[y.shape[0] // 2 - sample_rate * 50 : y.shape[0] // 2 + sample_rate * 50]
    sf.write("tempaudio.wav", y_seg, sample_rate)

    my_csv = extractfeature(y_seg, sr, my_3_csv=my_3_csv)
    df = pd.DataFrame(my_csv)

    X, df = readdata(df)
    pred_x = model.predict(X)
    pred_ind = np.argmax(pred_x, axis=1)
    print('predict with CNN: ', predict2label(pred_ind)[:1])
    if debug:
        print('predict with KNN: ', predict2label(knn.predict(X)))
        print('predict with DecisionTree: ', predict2label(dt.predict(X)))
        print('predict with RandomForest: ', predict2label(rf.predict(X)))


if __name__ == '__main__':
    main()
##    for sn in [s for s in os.listdir(os.path.join('C:', 'Users', 'hien.doan', 'Music') if s.endswith('.wav'))]:
##        main([os.path.join('C:', 'Users', 'hien.doan', 'Music', sn)])
