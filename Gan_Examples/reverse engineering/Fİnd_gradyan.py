import h5py
import numpy as np
import tensorflow as tf

# Eğitilmiş modelin h5 dosyasını açın
file = h5py.File("deneme.h5", "r")

# Modeli oluşturun
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=512, input_shape=(784,), name='dense'))  # Giriş şeklini uygun şekilde ayarlayın
model.add(tf.keras.layers.Dense(units=10, name='dense_1'))

# Ağırlıkları yüklerken şekilleri değiştirin
dense_weights = file["model_weights"]["dense"]["dense"]["kernel:0"][:]
dense_bias = file["model_weights"]["dense"]["dense"]["bias:0"][:]
dense_1_weights = file["model_weights"]["dense_1"]["dense_1"]["kernel:0"][:]
dense_1_bias = file["model_weights"]["dense_1"]["dense_1"]["bias:0"][:]

reshaped_weights = dense_weights.reshape((784, 512))  # Yeni şekil: (784, 512)
reshaped_bias = dense_bias.reshape((512,))  # Yeni şekil: (512,)

model.get_layer("dense").set_weights([reshaped_weights, reshaped_bias])
model.get_layer("dense_1").set_weights([dense_1_weights, dense_1_bias])

# Giriş verilerini hazırlayın (örnek olarak rastgele veri kullanalım)
input_dim = 784  # Giriş verilerinin boyutu
# input_data = np.random.rand(1, input_dim)  # 1 örneklik rastgele giriş verisi
input_data = tf.constant(input_data) 
# Giriş verilerini modele uygulayarak çıktıyı elde edin
with tf.GradientTape() as tape:
    tape.watch(input_data)
    output = model(input_data)

# Gradyanı hesaplayın
gradients = tape.gradient(output, input_data)

# Gradyanı yazdırın
print("Gradyanlar:", gradients)
