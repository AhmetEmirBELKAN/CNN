import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import cv2

# GAN modelini oluşturma
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_shape=(100,), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100 * 100, activation='tanh'))
    model.add(layers.Reshape((100, 100, 1)))
    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(100, 100, 1)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model
# Dog klasöründeki fotoğrafları yükleme
image_size = (100, 100)
image_folder = 'plane'
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
x_train = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size)
    img = img.reshape((*image_size, 1))
    img = (img - 127.5) / 127.5  # Normalize etme [-1, 1] aralığına
    x_train.append(img)
x_train = np.array(x_train)

# Model ve optimizasyon fonksiyonlarını oluşturma
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

loss_function = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# Eğitim döngüsü
batch_size = 128
epochs = 1000000000000
steps_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Gerçek veri örnekleri
        real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        
        # Gürültüden oluşturulan sahte veri örnekleri
        noise = np.random.normal(loc=0, scale=1, size=(batch_size, 100))
        generated_images = generator.predict(noise)
        
        # Etiketler
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Ayırt edici ağ eğitimi
        with tf.GradientTape() as tape:
            real_logits = discriminator(real_images)
            fake_logits = discriminator(generated_images)
            
            d_loss_real = loss_function(real_labels, real_logits)
            d_loss_fake = loss_function(fake_labels, fake_logits)
            d_loss = d_loss_real + d_loss_fake
        
        gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
        
        # Üretici ağ eğitimi
        noise = np.random.normal(loc=0, scale=1, size=(batch_size, 100))
        
        with tf.GradientTape() as tape:
            generated_images = generator(noise)
            fake_logits = discriminator(generated_images)
            
            g_loss = loss_function(real_labels, fake_logits)
        
        gradients = tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        
        # İlerleme çıktısı
        if (step+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{steps_per_epoch}] Discriminator Loss: {d_loss:.4f} Generator Loss: {g_loss:.4f}")
    
    # Her epoch sonunda üretilen resimleri kaydetme
    if (epoch+1) % 100 == 0:
        noise = np.random.normal(loc=0, scale=1, size=(10, 100))
        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5  # [-1, 1] aralığından [0, 1] aralığına çevirme
        for i in range(generated_images.shape[0]):
            image = generated_images[i, :, :, 0]
            image = np.uint8(image * 255)
            # cv2.imwrite(f"generated_image_{epoch+1}_{i+1}.png", image)
            cv2.imshow(f"Generated Image {epoch+1}_{i+1}", image)
        
        # Güncelleme ve klavye girişlerini kontrol etme
            key = cv2.waitKey(100)  # 1 milisaniye beklet
            if key == ord('q'):  # 'q' tuşuna basılırsa döngüyü sonlandır
                break
            cv2.destroyAllWindows()
# Eğitim tamamlandıktan sonra modeli kaydetme
generator.save("generator_model2.h5")
