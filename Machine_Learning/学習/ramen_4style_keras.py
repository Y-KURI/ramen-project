from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

# 分類対象のカテゴリ
root_dir = "./ramen_image(2)/"
styles =["miso", "sio", "tonkotsu", "shouyu"]
nb_classes = len(styles)
image_size = 50

# データをロード --- (※1)
def main():
    X_train, X_test, y_train, y_test = np.load("./ramen_image(2)/4style.npy")
    # データを正規化する
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float")  / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    # モデルを訓練し評価する
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

# モデルを構築 --- (※2)
def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
	border_mode='same',
#’valid’の場合、（フィルタのサイズやストライドにもよりますが）出力のサイズは入力に対して小さくなります。
#一方、’same’を指定するとゼロパディングが適用され、畳み込み層の入力と出力のサイズが同じになります（ストライドが1の場合）
	input_shape=in_shape))#input配列サイズ (samples, channels, rows, cols)の4次元テンソル
                          #output配列サイズ(samples, nb_filter, new_rows, new_cols)の4次元テンソル
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))#ドロップする割合
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())#入力を平坦化する．バッチサイズに影響されない．
    model.add(Dense(512))#全結合
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))#全結合
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])
    return model

# モデルを訓練する --- (※3)
def model_train(X, y):
    model = build_model(X.shape[1:])
    model.fit(X, y, batch_size=32, nb_epoch=30)
    # モデルを保存する --- (※4)
    hdf5_file = "./ramen_image(2)/ramen_model.hdf5"
    model.save_weights(hdf5_file)
    return model

# モデルを評価する --- (※5)
def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])

if __name__ == "__main__":
    main()
