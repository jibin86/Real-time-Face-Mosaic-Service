from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r34" ### 수정!
# config.network = "r18" ### 수정!
config.save_all_states = True
config.resume = True
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 16 ### 수정!
config.lr = 0.02
config.verbose = 2000
config.dali = False

config.rec = "/content/drive/MyDrive/딥러닝/RealTimeFaceRecognition/Pytorch_ArcFace/arcface_torch/train_tmp/ms1m-retinaface-t1"
config.num_classes = 165 ### 수정!
config.num_image = 37599 ### 수정!
config.num_epoch = 20 ### 수정!
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
