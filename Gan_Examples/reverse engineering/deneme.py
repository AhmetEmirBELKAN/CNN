import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import save_model

# MNIST veri setini yükleme
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Veri setini düzleştirme ve normalizasyon
X_train = X_train.reshape((60000, 784)).astype('float32') / 255
X_test = X_test.reshape((10000, 784)).astype('float32') / 255

# Etiketleri kategorik hale getirme
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modelin oluşturulması
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

# Modelin derlenmesi
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitimi
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

# Modeli h5 dosyası olarak kaydetme
save_model(model, 'egitilmis_model.h5')
