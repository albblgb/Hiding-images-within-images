import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch
import os
from PIL import Image
from torchvision.utils import save_image

from models.WengNet import wengnet
from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.image import calculate_psnr, calculate_ssim, calculate_mae, calculate_rmse
from utils.dataset import load_dataset
from utils.dirs import mkdirs
import config as c
from utils.model import load_model


os.environ["CUDA_VISIBLE_DEVICES"] = c.wengnet_device_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.join(c.model_dir, 'wengnet')
mkdirs(model_save_path)
train_data_dir = os.path.join(c.data_dir, c.data_name_train, 'train')
test_data_dir = os.path.join(c.data_dir, c.data_name_test, 'test')

mkdirs('results')
logger_name = 'wengnet_trained_ON_' + c.data_name_train
logger_info(logger_name, log_path=os.path.join('results', logger_name+'.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: wengnet')
logger.info('train data: {:s}'.format(c.data_name_train))
logger.info('test data: {:s}'.format(c.data_name_test))
logger.info('mode: {:s}'.format(c.mode))

train_loader, test_loader = load_dataset(train_data_dir, test_data_dir, c.wengnet_batch_size_train, c.wengnet_batch_size_test)

model = wengnet().to(device)
# model.load_state_dict(torch.load('model_zoo/wengnet/checkpoint_3000.pt'))

if c.mode == 'test':

    model.load_state_dict(torch.load(c.test_wengnet_path))

    with torch.no_grad():
        S_psnr = []; S_ssim = []; S_mae = []; S_rmse = []
        R_psnr = []; R_ssim = []; R_mae = []; R_rmse = []
    
        model.eval()
        stream = tqdm(test_loader)
        for idx, data in enumerate(stream):
            data = data.cuda()
            secret = data[data.shape[0]//2:]
            cover = data[:data.shape[0]//2]

            ################## forward ####################
            stego, secret_rev = model(secret, cover, 'test')
            
            cover_resi = abs(cover - stego) * 5
            secret_resi = abs(secret - secret_rev) * 5
            
            ############### save images #################
            if c.save_processed_img == True:
                super_dirs = ['cover', 'secret', 'stego', 'secret_rev', 'cover_resi', 'secret_resi']
                for cur_dir in super_dirs:
                    mkdirs(os.path.join(c.IMAGE_PATH, 'wengnet', c.data_name_test, cur_dir))    
                image_name = '%.4d.' % idx + c.suffix
                save_image(cover, os.path.join(c.IMAGE_PATH, 'wengnet', c.data_name_test, super_dirs[0], image_name))
                save_image(secret, os.path.join(c.IMAGE_PATH, 'wengnet', c.data_name_test, super_dirs[1], image_name))
                save_image(stego, os.path.join(c.IMAGE_PATH, 'wengnet', c.data_name_test, super_dirs[2], image_name))
                save_image(secret_rev, os.path.join(c.IMAGE_PATH, 'wengnet', c.data_name_test, super_dirs[3], image_name))
                save_image(cover_resi, os.path.join(c.IMAGE_PATH, 'wengnet', c.data_name_test, super_dirs[4], image_name))
                save_image(secret_resi, os.path.join(c.IMAGE_PATH, 'wengnet', c.data_name_test, super_dirs[5], image_name))

            secret_rev = secret_rev.detach().cpu().numpy().squeeze() * 255
            np.clip(secret_rev, 0, 255)
            secret = secret.detach().cpu().numpy().squeeze() * 255
            np.clip(secret, 0, 255)
            cover = cover.detach().cpu().numpy().squeeze() * 255
            np.clip(cover, 0, 255)
            stego = stego.detach().cpu().numpy().squeeze() * 255
            np.clip(stego, 0, 255)

            psnr_temp = calculate_psnr(cover, stego)
            S_psnr.append(psnr_temp)
            psnr_temp = calculate_psnr(secret, secret_rev)
            R_psnr.append(psnr_temp)

            mae_temp = calculate_mae(cover, stego)
            S_mae.append(mae_temp)
            mae_temp = calculate_mae(secret, secret_rev)
            R_mae.append(mae_temp)

            rmse_temp = calculate_rmse(cover, stego)
            S_rmse.append(rmse_temp)
            rmse_temp = calculate_rmse(secret, secret_rev)
            R_rmse.append(rmse_temp)

            ssim_temp = calculate_ssim(cover, stego)
            S_ssim.append(ssim_temp)
            ssim_temp = calculate_ssim(secret, secret_rev)
            R_ssim.append(ssim_temp)

        logger.info('testing, stego_avg_psnr: {:.2f}, secref_avg_psnr: {:.2f}'.format(np.mean(S_psnr), np.mean(R_psnr)))
        logger.info('testing, stego_avg_ssim: {:.4f}, secref_avg_ssim: {:.4f}'.format(np.mean(S_ssim), np.mean(R_ssim)))
        logger.info('testing, stego_avg_mae: {:.2f}, secref_avg_mae: {:.2f}'.format(np.mean(S_mae), np.mean(R_mae)))
        logger.info('testing, stego_avg_rmse: {:.2f}, secref_avg_rmse: {:.2f}'.format(np.mean(S_rmse), np.mean(R_rmse)))
else:
    secret_restruction_loss = nn.MSELoss().cuda()
    stego_similarity_loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)

    for epoch in range(c.epochs):
        epoch += 1
        s_loss = []
        r_loss = []
        loss_history=[]
        ###############################################################
        #                            train                            # 
        ###############################################################
        model.train()
        metric_monitor = MetricMonitor(float_precision=4)
        stream = tqdm(train_loader)

        for batch_idx, data in enumerate(stream):
            data = data.cuda()
            secret = data[data.shape[0]//2:]
            cover = data[:data.shape[0]//2]
            
            ################## forward ####################
            stego, secret_rev = model(secret, cover)

            ################### loss ######################
            S_loss = stego_similarity_loss(cover, stego)
            R_loss = secret_restruction_loss(secret, secret_rev)
            loss =  c.wengnet_lambda_S * S_loss + c.wengnet_lambda_R * R_loss

            ################### backword ##################
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            s_loss.append(S_loss.item())
            r_loss.append(R_loss.item())
            loss_history.append(loss.item())

            metric_monitor.update("S_loss", np.mean(np.array(s_loss)))
            metric_monitor.update("R_loss", np.mean(np.array(r_loss)))
            metric_monitor.update("T_Loss", np.mean(np.array(loss_history)))
            stream.set_description(
                "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        epoch_losses = np.mean(np.array(loss_history))

        ###############################################################
        #                              val                            # 
        ###############################################################
        model.eval()
        if epoch % c.test_freq == 0:
            with torch.no_grad():
                S_psnr = []
                R_psnr = []
                for data in test_loader:
                    data = data.cuda()
                    secret = data[data.shape[0]//2:]
                    over = data[:data.shape[0]//2]

                    ################## forward ####################
                    stego, secret_rev = model(secret, cover)

                    ############### calculate psnr #################
                    secret_rev = secret_rev.detach().cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.detach().cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.detach().cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    stego = stego.detach().cpu().numpy().squeeze() * 255
                    np.clip(stego, 0, 255)
                    psnr_temp = calculate_psnr(cover, stego)
                    S_psnr.append(psnr_temp)
                    psnr_temp = calculate_psnr(secret, secret_rev)
                    R_psnr.append(psnr_temp)
                logger.info('epoch: {}, training,  loss: {}'.format(epoch, epoch_losses))
                logger.info('epoch: {}, testing, stego_avg_psnr: {:.2f}, secref_avg_psnr: {:.2f}'.format(epoch, np.mean(S_psnr), np.mean(R_psnr)))

        if epoch % c.save_freq == 0 and epoch >= c.save_start_epoch:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'checkpoint_%.4i' % epoch + '.pt'))
            
        scheduler.step()


    
