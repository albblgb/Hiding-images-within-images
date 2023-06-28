import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import logging
import numpy as np
import math
from torchvision.utils import save_image

from models.HiDDeN import EncoderDecoder, Discriminator
from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.image import calculate_psnr, calculate_ssim, calculate_mae, calculate_rmse
from utils.dataset import load_dataset
from utils.dirs import mkdirs
import config as c
from utils.model import load_model


os.environ["CUDA_VISIBLE_DEVICES"] = c.hidden_device_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.join(c.model_dir, 'hidden')
mkdirs(model_save_path)
train_data_dir = os.path.join(c.data_dir, c.data_name_train, 'train')
test_data_dir = os.path.join(c.data_dir, c.data_name_test, 'test')

mkdirs('results')
logger_name = 'hidden_trained_ON_' + c.data_name_train
logger_info(logger_name, log_path=os.path.join('results', logger_name+'.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: hidden')
logger.info('train data: {:s}'.format(c.data_name_train))
logger.info('test data: {:s}'.format(c.data_name_test))
logger.info('mode: {:s}'.format(c.mode))

################## prepare ####################
enc_decoder = EncoderDecoder().to(device)
discriminator = Discriminator().to(device)
# model = nn.DataParallel(model)
train_loader, test_loader = load_dataset(train_data_dir, test_data_dir, c.hidden_batch_size_train, c.hidden_batch_size_test)

if c.mode == 'test':
    # enc_decoder = torch.load(c.test_hidden_path)
    enc_decoder.load_state_dict(torch.load(c.test_hidden_path))
    
    with torch.no_grad():
        S_psnr = []; S_ssim = []; S_mae = []; S_rmse = []
        R_psnr = []; R_ssim = []; R_mae = []; R_rmse = []

        enc_decoder.eval()
        stream = tqdm(test_loader)
        for idx, data in enumerate(stream):
            data = data.to(device)
            secret = data[data.shape[0]//2:]
            cover = data[:data.shape[0]//2]

            ################## forward ####################
            stego, secret_rev = enc_decoder(cover, secret)
            
            cover_resi = abs(cover - stego) * 5
            secret_resi = abs(secret - secret_rev) * 20
            
            ############### save images #################
            if c.save_processed_img == True:
                super_dirs = ['cover', 'secret', 'stego', 'secret_rev', 'cover_resi', 'secret_resi']
                for cur_dir in super_dirs:
                    mkdirs(os.path.join(c.IMAGE_PATH, 'hidden', c.data_name_test, cur_dir))    
                image_name = '%.4d.' % idx + c.suffix
                save_image(cover, os.path.join(c.IMAGE_PATH, 'hidden', c.data_name_test, super_dirs[0], image_name))
                save_image(secret, os.path.join(c.IMAGE_PATH, 'hidden', c.data_name_test, super_dirs[1], image_name))
                save_image(stego, os.path.join(c.IMAGE_PATH, 'hidden', c.data_name_test, super_dirs[2], image_name))
                save_image(secret_rev, os.path.join(c.IMAGE_PATH, 'hidden', c.data_name_test, super_dirs[3], image_name))
                save_image(cover_resi, os.path.join(c.IMAGE_PATH, 'hidden', c.data_name_test, super_dirs[4], image_name))
                save_image(secret_resi, os.path.join(c.IMAGE_PATH, 'hidden', c.data_name_test, super_dirs[5], image_name))

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

    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
    mse_loss = nn.MSELoss().to(device)

    optimizer_enc_dec = torch.optim.Adam(enc_decoder.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    optimizer_discrim = torch.optim.Adam(discriminator.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    scheduler_enc_dec = torch.optim.lr_scheduler.StepLR(optimizer_enc_dec, c.weight_step, gamma=c.gamma) 
    scheduler_discrim = torch.optim.lr_scheduler.StepLR(optimizer_discrim, c.weight_step, gamma=c.gamma)  

    cover_label = 1
    stego_label = 0

    for epoch in range(c.epochs):
        epoch += 1
        loss_history=[]
        s_loss_history = []
        r_loss_history = []
        a_loss_history = []
        dc_loss_history = []
        ds_loss_history = []
        ###############################################################
        #                            train                            # 
        ###############################################################
        enc_decoder.train()
        discriminator.train()
        metric_monitor = MetricMonitor(float_precision=4)
        stream = tqdm(train_loader)

        for batch_idx, data in enumerate(stream):
            data = data.to(device)
            secret = data[data.shape[0]//2:]
            cover = data[:data.shape[0]//2]
            batch_size_half = secret.shape[0]

            d_target_label_cover = torch.full((batch_size_half, 1), cover_label, device=device).float()
            d_target_label_stego = torch.full((batch_size_half, 1), stego_label, device=device).float()
            g_target_label_stego = torch.full((batch_size_half, 1), cover_label, device=device).float()

            ################## train the discriminator ####################
            optimizer_discrim.zero_grad()
            # train on cover
            d_on_cover = discriminator(cover)
            d_loss_on_cover = bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()

            # train on stego
            stego, secret_rev = enc_decoder(cover, secret)
            d_on_stego = discriminator(stego.detach())
            d_loss_on_stego = bce_with_logits_loss(d_on_stego, d_target_label_stego)
            d_loss_on_stego.backward()

            optimizer_discrim.step()
            
            ################### Train the generator (encoder-decoder) ######################
            optimizer_enc_dec.zero_grad()
            # target label for stego images should be 'cover', because we want to fool the discriminator
            d_on_stego_for_enc = discriminator(stego)
            g_loss_adv = bce_with_logits_loss(d_on_stego_for_enc, g_target_label_stego)
            g_loss_enc = mse_loss(stego, cover)
            g_loss_dec = mse_loss(secret_rev, secret)
            g_loss = c.hidden_lambda_Adv * g_loss_adv + c.hidden_lambda_S * g_loss_enc + c.hidden_lambda_R * g_loss_dec

            g_loss.backward()
            optimizer_enc_dec.step()
            
            s_loss_history.append([g_loss_enc.item(), 0.])
            r_loss_history.append([g_loss_dec.item(), 0.])
            a_loss_history.append([g_loss_adv.item(), 0.])
            dc_loss_history.append([d_loss_on_cover.item(), 0.])
            ds_loss_history.append([d_loss_on_stego.item(), 0.])
            loss_history.append(g_loss.item()+d_loss_on_cover.item()+d_loss_on_stego.item())

            metric_monitor.update("S_loss", np.array(s_loss_history).mean()) # similarity loss between cover and stego
            metric_monitor.update("R_loss", np.array(r_loss_history).mean()) # recover loss between secret and secret_rev
            metric_monitor.update("A_Loss", np.array(a_loss_history).mean()) # adv loss of generator
            metric_monitor.update("DC_Loss", np.array(dc_loss_history).mean()) # the loss of discrimitor on cover
            metric_monitor.update("DS_Loss", np.array(ds_loss_history).mean()) # the loss of discrimitor on stego
            stream.set_description(
                "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        epoch_losses = np.mean(np.array(loss_history))

        ###############################################################
        #                              val                            # 
        ###############################################################
        enc_decoder.eval()
        # discriminator.eval()
        if epoch % c.test_freq == 0:
            with torch.no_grad():
                S_psnr = []
                R_psnr = []
                for data in test_loader:
                    data = data.to(device)
                    secret = data[data.shape[0]//2:]
                    cover = data[:data.shape[0]//2]

                    ################## forward ####################
                    stego, secret_rev = enc_decoder(cover, secret)

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
            # we only save the encoder and decoder of hidden.
            torch.save(enc_decoder.state_dict(), os.path.join(model_save_path, 'checkpoint_%.4i' % epoch + '.pt'))
            
        scheduler_enc_dec.step()
        scheduler_discrim.step()


        
