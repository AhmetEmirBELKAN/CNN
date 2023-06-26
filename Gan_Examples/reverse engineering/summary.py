import tensorflow as tf

model = tf.keras.models.Sequential()

# Dense katmanını ekleme
input_dim = 100  # Giriş verilerinin boyutu
model.add(tf.keras.layers.Dense(units=512, input_shape=(input_dim,), name='dense'))
model.add(tf.keras.layers.Dense(units=10, name='dense_1'))

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli özetleme
model.summary()
