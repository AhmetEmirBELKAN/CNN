import tensorflow as tf
import h5py

# Eğitilmiş modelin h5 dosyasını açın
file = h5py.File("deneme.h5", "r")

# 'model_weights' grubunu kontrol etme
if "model_weights" in file:
    model_weights_group = file["model_weights"]
    # Elemanlara tek tek erişim
    for member_name in model_weights_group.keys():
        member = model_weights_group[member_name]
        print("Eleman Adı:", member_name)
        print("Eleman Değeri:")
        for submember_name in member.keys():
            submember = member[submember_name]
            print(submember_name, ":", submember[next(iter(submember))])  # Değerleri doğrudan yazdırın
        
        # Ağırlıkların şeklini ve boyutunu alın
        if member_name == "dense":
            dense_weights = member["dense"]["kernel:0"][()]
            print("Ağırlıkların Şekli:", dense_weights.shape)
            print("Ağırlıkların Boyutu:", dense_weights.size)
        
        print()

else:
    print("'model_weights' grubu bulunamadı.")

# Dosyayı kapatın
file.close()
