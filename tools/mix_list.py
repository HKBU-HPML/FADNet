import numpy as np

train_list = []
test_list = []

mix_train_file = open("mix_sgm_release_TRAIN.list", "w")
mix_test_file = open("mix_sgm_release_TEST.list", "w")

# read flying original disparity
f = open("FlyingThings3D_release_TRAIN.list")
flying_train = f.readlines()
f.close()

f = open("FlyingThings3D_release_TEST.list")
flying_test = f.readlines()
f.close()

# read flying sgm disparity
f = open("FlyingThings3D_sgm_release_TRAIN.list")
flying_sgm_train = f.readlines()
f.close()

f = open("FlyingThings3D_sgm_release_TEST.list")
flying_sgm_test = f.readlines()
f.close()

# read real sgm disparity
f = open("real_sgm_release.list")
real_sgm = f.readlines()
f.close()

# pick 100 training sample from flying original and sgm training
rand_idx = np.random.randint(len(flying_train), size = 50)
print rand_idx
for idx in rand_idx:
    train_list.append(flying_train[idx])
    train_list.append(flying_train[idx].replace("disparity", "sgm_disp"))

# pick 70 training sample from real sgm
train_list.extend(real_sgm)
train_list.extend(real_sgm)

# pick 50 testing sample from flying original and sgm testing
rand_idx = np.random.randint(len(flying_test), size = 25)
print rand_idx
for idx in rand_idx:
    test_list.append(flying_test[idx])
    test_list.append(flying_test[idx].replace("disparity", "sgm_disp"))

# pick 21 testing sample from real sgm
test_list.extend(real_sgm[70:])

for item in train_list:
    mix_train_file.write(item)
mix_train_file.close()

for item in test_list:
    mix_test_file.write(item)
mix_test_file.close()

