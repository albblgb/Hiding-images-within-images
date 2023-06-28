import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import logging
import numpy as np
from torchvision.utils import save_image

from models.PUSNet import pusnet
from utils.logger import logger_info
from utils.image import calculate_psnr, calculate_ssim, calculate_mae, calculate_rmse
from utils.dataset import load_dataset
from utils.dirs import mkdirs
from utils.model import load_model
import config as c
from utils.proposed_mothod import generate_sparse_mask, init_weights, remove_adapter, insert_adapter


parser = argparse.ArgumentParser()
parser.add_argument('--sr', default=0.1, type=float, help='sparse ratio')   
parser.add_argument('--mt', default='lwf', type=str, help='the type of mask')  
parser.add_argument('--ds', default='div2k', type=str, help='dataset: div2k, imagenet, coco')   
parser.add_argument('--ts', default='hiding', type=str, help='test state, hiding or recover')   
args = parser.parse_args()

ATTACK_TIMES = 100
ra_psnr = []
ra_apd = []

# os.environ["CUDA_VISIBLE_DEVICES"] = c.pusnet_device_ids
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data_dir = os.path.join(c.data_dir, c.data_name_train, 'train')
test_data_dir = os.path.join(c.data_dir, args.ds, 'test')

logger_name = 'random_guess_attack_on_sparse_USNet'
logger_info(logger_name, log_path=os.path.join('./results', logger_name+'.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: pusnet')
logger.info('train data: {:s}'.format(c.data_name_train))
logger.info('test data: {:s}'.format(args.ds))
logger.info('mode: {:s}'.format(c.mode))
logger.info('sparse ration: {:s}'.format(str(args.sr)))
logger.info('test state: {:s}'.format(str(args.ts)))
logger.info('correct k_h/k_r: {:s}'.format('10101/1010'))

model_hiding_seed = pusnet()
model_recover_seed = pusnet()
# mask generation accoding to random seed '1'
init_weights(model_hiding_seed, random_seed=1)
sparse_mask = generate_sparse_mask(model_hiding_seed, sparse_ratio=args.sr, sc=args.mt)
for idx in range(len(sparse_mask)):
    sparse_mask[idx] = sparse_mask[idx].to(device)

model = pusnet().to(device)
# init_weights(model)
model_hiding_seed = model_hiding_seed.to(device)
model_recover_seed = model_recover_seed.to(device)

# multi GPUs
model = nn.DataParallel(model)
model_hiding_seed = nn.DataParallel(model_hiding_seed)
model_recover_seed = nn.DataParallel(model_recover_seed)

_, test_loader = load_dataset(train_data_dir, test_data_dir, c.pusnet_batch_size_train, c.pusnet_batch_size_test, c.pusnet_sigma)

model.load_state_dict(torch.load(c.test_pusnet_path))

for i in range(ATTACK_TIMES):
    
    MAXSEED = 1000000
    if args.ts == 'hiding':
        guessed_k_h = np.random.randint(0,MAXSEED)   # when test hiding state, use random guessed hiding key
    else:
        guessed_k_h = 10101     # ramdom guessed hiding key, use correct hiding key but random guessed recover key
    guessed_k_r = np.random.randint(0,MAXSEED)
    logger.info('*'*50)
    logger.info('guessed_k_h: {:s}'.format(str(guessed_k_h)))
    logger.info('guessed_k_r: {:s}'.format(str(guessed_k_r)))

    # set hiding seed/key '10101' and recover seed/key '1010'
    init_weights(model_hiding_seed, random_seed=guessed_k_h)
    init_weights(model_recover_seed, random_seed=guessed_k_r)

    with torch.no_grad():
        S_psnr = []; S_ssim = []; S_mae = []; S_rmse = []
        R_psnr = []; R_ssim = []; R_mae = []; R_rmse = []
        model.eval()

        if args.ts == 'hiding':
            insert_adapter(model, sparse_mask, model_hiding_seed, is_sparse=False)
            stream = tqdm(test_loader)
            for idx, (data, noised_data) in enumerate(stream):
                data = data.to(device)
                secret = data[data.shape[0]//2:]
                cover = data[:data.shape[0]//2]

                ################## forward ####################
                stego = model(secret, cover, 'hiding')

                ############### calculate metrics #################
                secret = secret.detach().cpu().numpy().squeeze() * 255
                np.clip(secret, 0, 255)
                cover = cover.detach().cpu().numpy().squeeze() * 255
                np.clip(cover, 0, 255)
                stego = stego.detach().cpu().numpy().squeeze() * 255
                np.clip(stego, 0, 255)
                
                psnr_temp = calculate_psnr(cover, stego)
                S_psnr.append(psnr_temp)

                mae_temp = calculate_mae(cover, stego)
                S_mae.append(mae_temp)

                rmse_temp = calculate_rmse(cover, stego)
                S_rmse.append(rmse_temp)

                ssim_temp = calculate_ssim(cover, stego)
                S_ssim.append(ssim_temp)

            logger.info('testing, stego_avg_psnr: {:.2f}'.format(np.mean(S_psnr)))
            logger.info('testing, stego_avg_ssim: {:.4f}'.format(np.mean(S_ssim)))
            logger.info('testing, stego_avg_mae: {:.2f}'.format(np.mean(S_mae)))
            logger.info('testing, stego_avg_rmse: {:.2f}'.format(np.mean(S_rmse)))

        else:
            stream = tqdm(test_loader)
            for idx, (data, noised_data) in enumerate(stream):
                data = data.to(device)

                secret = data[data.shape[0]//2:]
                cover = data[:data.shape[0]//2]

                ################## forward ####################
                insert_adapter(model, sparse_mask, model_hiding_seed, is_sparse=False)
                stego = model(secret, cover, 'hiding')
                insert_adapter(model, sparse_mask, model_recover_seed, is_sparse=False)
                secret_rev = model(stego, None, 'recover')

                ############### calculate metrics #################
                secret = secret.detach().cpu().numpy().squeeze() * 255
                np.clip(secret, 0, 255)
                secret_rev = secret_rev.detach().cpu().numpy().squeeze() * 255
                np.clip(secret_rev, 0, 255)

                psnr_temp = calculate_psnr(secret, secret_rev)
                R_psnr.append(psnr_temp)

                mae_temp = calculate_mae(secret, secret_rev)
                R_mae.append(mae_temp)

                rmse_temp = calculate_rmse(secret, secret_rev)
                R_rmse.append(rmse_temp)

                ssim_temp = calculate_ssim(secret, secret_rev)
                R_ssim.append(ssim_temp)
            
            logger.info('testing, secref_avg_psnr: {:.2f}'.format(np.mean(R_psnr)))
            logger.info('testing, secref_avg_ssim: {:.4f}'.format(np.mean(R_ssim)))
            logger.info('testing, secref_avg_mae: {:.2f}'.format(np.mean(R_mae)))
            logger.info('testing, secref_avg_rmse: {:.2f}'.format(np.mean(R_rmse)))
        if args.ts == 'hiding': 
            ra_psnr.append(np.mean(S_psnr))
            ra_apd.append(np.mean(S_mae))
        else:
            ra_psnr.append(np.mean(R_psnr))
            ra_apd.append(np.mean(R_mae))            

ra_psnr = np.array(ra_psnr)
ra_apd = np.array(ra_apd)

logger.info('#'*50)
logger.info('final_result: {:s} on {:s}'.format(args.ts, args.ds))
logger.info('psnr mean: {:.2f},  std: {:.2f}'.format(np.mean(ra_psnr), np.std(ra_psnr)))
logger.info('psnr mean: {:.2f},  std: {:.2f}'.format(np.mean(ra_apd), np.std(ra_apd)))