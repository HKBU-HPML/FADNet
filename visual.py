from dataset import *
import matplotlib.pyplot as plt
from torchvision import transforms, utils

disp_dataset = DispDataset(txt_file = 'FlyingThings3D_release_TRAIN.list', root_dir = 'data')

fig = plt.figure()
scale = RandomRescale((1024, 1024))
crop = RandomCrop((384, 768))
composed = transforms.Compose([scale, crop])
tsfrm = [scale, crop, composed]

show_num = 3
# for i in range(show_num):
for i in range(len(tsfrm)):

    sample = disp_dataset[0]
    sample = tsfrm[i](sample)

    print(i, sample['img_left'].shape,  \
             sample['img_right'].shape, \
             # sample['pm_disp'].shape,    \
             # sample['pm_cost'].shape,    \
             sample['gt_disp'].shape  \
         )

    ax = plt.subplot(show_num, 3, i * 3 + 1)
    # plt.tight_layout()
    ax.set_title('Sample #{} img_left'.format(i))
    ax.axis('off')
    plt.imshow(sample['img_left'])
    # plt.pause(0.001)

    ax = plt.subplot(show_num, 3, i * 3 + 2)
    # plt.tight_layout()
    ax.set_title('Sample #{} img_right'.format(i))
    ax.axis('off')
    plt.imshow(sample['img_right'])
    # plt.pause(0.001)

    ax = plt.subplot(show_num, 3, i * 3 + 3)
    # plt.tight_layout()
    ax.set_title('Sample #{} gt_disp'.format(i))
    ax.axis('off')
    plt.imshow(sample['gt_disp'], cmap='gray')
    # plt.pause(0.001)


plt.show()

