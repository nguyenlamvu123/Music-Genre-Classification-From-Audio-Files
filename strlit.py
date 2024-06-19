from coordinate_constant import extractfeature, readdata, model, converter, songname, header
import librosa, csv
import numpy as np
import pandas as pd


y, sr = librosa.load(songname, mono=True, duration=30)
to_append = extractfeature(y, sr, filename='_', g='_')

file = open('try_running__Extracted___Data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerow(to_append.split())

df = pd.read_csv('try_running__Extracted___Data.csv')
X, df = readdata(df)
pred_x = model.predict(X)
pred_ind = np.argmax(pred_x, axis=1)
print(converter.inverse_transform(pred_ind)[:1])