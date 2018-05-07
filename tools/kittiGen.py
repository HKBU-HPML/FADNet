import os

dataset = 'KITTI'

data_root = '%s_release' % dataset
trainf = open('%s_TRAIN.list' % dataset, 'w')
testf = open('%s_TEST.list' % dataset, 'w')

def rnn_walk(path):

    left_fold  = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'
  
    # train
    filepath = os.path.join(path, 'training')
    image = [img for img in os.listdir(os.path.join(filepath, left_fold)) if img.find('_10') > -1]
   
    left_train  = [filepath+left_fold+img for img in image]
    right_train = [filepath+right_fold+img for img in image]
    disp_train_L = [filepath+disp_L+img for img in image]

    for i in range(len(left_train)):
        trainf.write("%s %s %s\n" % (left_train[i], right_train[i], disp_train_L[i]))

    # test
    filepath = os.path.join(path, 'testing')
    image = [img for img in os.listdir(os.path.join(filepath, left_fold)) if img.find('_10') > -1]
   
    left_test  = [filepath+left_fold+img for img in image]
    right_test = [filepath+right_fold+img for img in image]
    disp_test_L = [filepath+disp_L+img for img in image]

    for i in range(len(left_test)):
        testf.write("%s %s %s\n" % (left_test[i], right_test[i], disp_test_L[i]))

rnn_walk(data_root)
trainf.close()
testf.close()





