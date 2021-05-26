import torch
from networks.FADNet import FADNet

net = FADNet()
model_data = torch.load('models/fadnet-sceneflow-4e-3-dynamic/model_best.pth')

source_sd = model_data['state_dict']
target_sd = net.state_dict()

for para in source_sd:
    if para in target_sd:
        target_sd[para] = source_sd[para]
        print(para, '->', para)
    new_para = para.replace('dispnetc', 'extract_network')
    if new_para in target_sd:
        target_sd[new_para] = source_sd[para]
        print(para, '->', new_para)
    new_para = para.replace('dispnetc', 'cunet')
    if new_para in target_sd:
        target_sd[new_para] = source_sd[para]
        print(para, '->', new_para)

net.load_state_dict(target_sd)
state = {'round': 4, 'epoch': 30, 'arch': 'dispnet', 'state_dict':net.state_dict(), 'best_EPE': 0.76}
torch.save(state, 'new.pth')



