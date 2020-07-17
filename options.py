gpu = '9'
random_seed = 0
mode = 1    # 1-rgb images, 2-thermal images, 3-audio, 4-rgb and thermal images, 8-all 

#dataset options
sub_path = 'data/subjects.csv'
train_path = 'datasets/SpeakingFaces/train_data'
valid_path = 'datasets/SpeakingFaces/valid_data'
#train_path = 'datasets/SpeakingFaces/valid_data'
#valid_path = 'datasets/SpeakingFaces/test_data'
test_path = 'datasets/SpeakingFaces/test_data'

#image file options
num_frames = 5
add_rgb_noise = True
rgb_noise = 'gauss'   # 'gauss', 
rsnr = 50.0    # signal to noise ration

#audio file options
segment_len = 0.2   # seconds
sample_rate = 44100 # audio file sampling rate
add_audio_noise = True
audio_noise = 'gauss'   # 'gauss', 
asnr = 0.005    # signal to noise ration

#model options
batch_size = 128
#base_lr = 1e-0 #audio
#base_lr = 1e-1 #image
base_lr = 1e-1
is_clip = True
clip = 10
weight_decay = 0.0
drop = 0.0
patience = 20
num_workers = 0
max_epoch = 200
save_prefix = f'models/mmnet'
data_shuffle = False
is_optimize = True

##LOAD PRETRAINED MODELS
#weights = 'pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
