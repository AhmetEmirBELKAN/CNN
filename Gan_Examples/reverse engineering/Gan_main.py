import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Eğitilmiş generator modelini yükle
generator = load_model('generator_model.h5')

# Yeni köpek resimleri oluştur
noise = np.random.normal(loc=0, scale=1, size=(10, 100))
generated_images = generator.predict(noise)
generated_images = 0.5 * generated_images + 0.5  # [-1, 1] aralığından [0, 1] aralığına çevirme

# Oluşturulan resimleri kaydet
for i in range(generated_images.shape[0]):
    image = generated_images[i, :, :, 0]
    image = np.uint8(image * 255)
    cv2.imwrite(f"generated_image_{i+1}.png", image)