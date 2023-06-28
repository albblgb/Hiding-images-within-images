##########################################################################
##                           shared super-parame                        ##
##########################################################################
mode = 'test' # train or test

# optim Adam
lr = 1e-4
epochs = 3000
weight_decay = 1e-5
weight_step = 500
betas = (0.5, 0.999)
gamma = 0.5

# dataset
crop_size_train = 256  # size for training
resize_size_test = 512  # size for testing
data_dir = '/data/gbli/gbData'
data_name_train = 'div2k'
data_name_test = 'div2k'

# Saving checkpoints
test_freq = 50
save_freq = 50
save_start_epoch = 1500
model_dir = 'model_zoo'

# Saving processed images
save_processed_img = True
IMAGE_PATH = 'results/images'
suffix = 'png'


##########################################################################
##             the paths of well trained models (for test)              ##
##########################################################################
test_pusnet_path = 'tmp/pusnet_checkpoint_2150.pt'
test_pusnet_p_path = 'model_zoo/pusnet_p/v2/checkpoint_2950.pt'
test_balujanet_path = 'tmp/baluja_checkpoint_3000.pt'
test_hidden_path = 'tmp/hidden_checkpoint_2850.pt'
test_wengnet_path = 'tmp/weng_checkpoint_2975.pt'
test_hinet_path = 'tmp/hinet_checkpoint_2500.pt'


##########################################################################
##                               pusnet                                ##
##########################################################################
pusnet_device_ids = '0, 1, 2, 3'
pusnet_batch_size_train = 8
pusnet_batch_size_test = 2

pusnet_sigma = 20

pusnet_lambda_S = 1.0
pusnet_lambda_R = 0.75
pusnet_lambda_DN = 0.25

pusnet_p_device_ids = '3, 4'
pusnet_p_batch_size_train = 8
pusnet_p_batch_size_test = 2

##########################################################################
##                               balujanet                              ##
##########################################################################
baluja_device_ids = '0'
baluja_batch_size_train = 8
baluja_batch_size_test = 2

baluja_lambda_S = 1.0
baluja_lambda_R = 0.75


##########################################################################
##                                 HiddeN                               ##
##########################################################################

hidden_device_ids = '1'
hidden_batch_size_train = 16
hidden_batch_size_test = 2

hidden_lambda_S = 0.75
hidden_lambda_R = 1.0
hidden_lambda_Adv = 1e-3


##########################################################################
##                                wengnet                               ##
##########################################################################

wengnet_device_ids = '2'
wengnet_batch_size_train = 16
wengnet_batch_size_test = 2

wengnet_lambda_S = 1.0
wengnet_lambda_R = 0.75

##########################################################################
##                                  HiNet                               ##
##########################################################################

hinet_device_ids = '0'
hinet_batch_size_train = 8
hinet_batch_size_test= 2

# Saving checkpoints:
hinet_test_freq = 20
hinet_save_freq = 20
# super-params of loss
lamda_reconstruction = 0.75
lamda_guide = 1
lamda_low_frequency = 0.25

channels_in = 3

