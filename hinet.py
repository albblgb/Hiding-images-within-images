import torch
import torch.nn as nn
import torch.optim
import math
import numpy as np
import logging
import argparse
import os
from tqdm import tqdm
from torchvision.utils import save_image

from utils.terminal import MetricMonitor
from utils.logger import logger_info
from utils.image import IWT, DWT, gauss_noise, calculate_psnr, quantization, calculate_ssim, calculate_mae, calculate_rmse
from utils.dataset import load_dataset
from models.HiNet import Model, init_model
from utils.dirs import mkdirs
from utils.model import load_model
import config as c


os.environ["CUDA_VISIBLE_DEVICES"] = c.hinet_device_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.join(c.model_dir, 'hinet')
mkdirs(model_save_path)
train_data_dir = os.path.join(c.data_dir, c.data_name_train, 'train')
test_data_dir = os.path.join(c.data_dir, c.data_name_test, 'test')

mkdirs('results')
logger_name = 'hinet_trained_ON_' + c.data_name_train
logger_info(logger_name, log_path=os.path.join('results', logger_name+'.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: hinet')
logger.info('train data: {:s}'.format(c.data_name_train))
logger.info('test data: {:s}'.format(c.data_name_test))
logger.info('mode: {:s}'.format(c.mode))


train_loader, test_loader = load_dataset(train_data_dir, test_data_dir, c.hinet_batch_size_train, c.hinet_batch_size_test)
dwt = DWT()
iwt = IWT()

net = Model().cuda()
if c.mode == 'test':
    net = load_model(net, c.test_hinet_path)

    with torch.no_grad():
        S_psnr = []; S_ssim = []; S_mae = []; S_rmse = []
        R_psnr = []; R_ssim = []; R_mae = []; R_rmse = []
        N_psnr = []; N_ssim = []; N_mae = []; N_rmse = []
        DN_psnr = []; DN_ssim = []; DN_mae = []; DN_rmse = []
        
        net.eval()
        stream = tqdm(test_loader)
        for idx, x in enumerate(stream):
            x = x.to(device)
            secret = x[x.shape[0] // 2:, :, :, :]
            cover = x[:x.shape[0] // 2, :, :, :]
            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_stego = output.narrow(1, 0, 4 * c.channels_in)        
            stego = iwt(output_stego) 
            stego = quantization(stego)
            output_stego = dwt(stego)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            output_z = gauss_noise(output_z.shape)

            #################
            #   backward:   #
            #################
            output_stego = output_stego.cuda()
            output_rev = torch.cat((output_stego, output_z), 1)
            output_image = net(output_rev, rev=True)
            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)
            secret_rev = quantization(secret_rev)

            cover_resi = abs(cover - stego) * 5
            secret_resi = abs(secret - secret_rev) * 5
            
            ############### save images #################
            if c.save_processed_img == True:
                super_dirs = ['cover', 'secret', 'stego', 'secret_rev', 'cover_resi', 'secret_resi']
                for cur_dir in super_dirs:
                    mkdirs(os.path.join(c.IMAGE_PATH, 'hinet', c.data_name_test, cur_dir))    
                image_name = '%.4d.' % idx + c.suffix
                save_image(cover, os.path.join(c.IMAGE_PATH, 'hinet', c.data_name_test, super_dirs[0], image_name))
                save_image(secret, os.path.join(c.IMAGE_PATH, 'hinet', c.data_name_test, super_dirs[1], image_name))
                save_image(stego, os.path.join(c.IMAGE_PATH, 'hinet', c.data_name_test, super_dirs[2], image_name))
                save_image(secret_rev, os.path.join(c.IMAGE_PATH, 'hinet', c.data_name_test, super_dirs[3], image_name))
                save_image(cover_resi, os.path.join(c.IMAGE_PATH, 'hinet', c.data_name_test, super_dirs[4], image_name))
                save_image(secret_resi, os.path.join(c.IMAGE_PATH, 'hinet', c.data_name_test, super_dirs[5], image_name))

            ############### calculate metrics #################
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
    init_model(net)
    net = torch.nn.DataParallel(net)

    guide_loss = nn.MSELoss().to(device)
    reconstruction_loss = nn.MSELoss().to(device)
    low_frequency_loss = nn.MSELoss().to(device)

    optim = torch.optim.Adam(net.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + 1 

        loss_history = []
        g_loss_history = []
        r_loss_history = []
        l_loss_history = []

        #################
        #     train:    #
        #################
        metric_monitor = MetricMonitor(float_precision=6)
        stream = tqdm(train_loader)

        for i_batch, data in enumerate(stream):
            data = data.to(device)
            secret = data[data.shape[0] // 2:]
            cover = data[:data.shape[0] // 2]
            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_stego = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            stego_img = iwt(output_stego)

            #################
            #   backward:   #
            #################

            output_z_guass = gauss_noise(output_z.shape)

            output_rev = torch.cat((output_stego, output_z_guass), 1)
            output_image = net(output_rev, rev=True)
            output_image = net(output_rev, rev=True)

            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)

            #################
            #     loss:     #
            #################
            g_loss = guide_loss(stego_img.cuda(), cover.cuda())
            r_loss = reconstruction_loss(secret_rev, secret)
            stego_low = output_stego.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(stego_low, cover_low)

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + min(i_epoch//2000, 1.0) * c.lamda_low_frequency * l_loss  
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            g_loss_history.append([g_loss.item(), 0.])
            r_loss_history.append([r_loss.item(), 0.])
            l_loss_history.append([l_loss.item(), 0.])
        
            metric_monitor.update("S_loss", np.array(g_loss_history).mean()) # similarity loss between cover and stego
            metric_monitor.update("R_loss", np.array(r_loss_history).mean()) # recover loss between secret and secret_rev
            metric_monitor.update("F_Loss", np.array(l_loss_history).mean()) # freq loss
            metric_monitor.update("T_Loss", np.array(loss_history).mean()) # total loss
            stream.set_description(
                "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=i_epoch, metric_monitor=metric_monitor)
            )

        #################
        #     val:    #
        #################
        if i_epoch % c.hinet_test_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                net.eval()
                for x in test_loader:
                    x = x.to(device)
                    secret = x[x.shape[0] // 2:, :, :, :]
                    cover = x[:x.shape[0] // 2, :, :, :]
                    cover_input = dwt(cover)
                    secret_input = dwt(secret)

                    input_img = torch.cat((cover_input, secret_input), 1)

                    #################
                    #    forward:   #
                    #################
                    output = net(input_img)
                    output_stego = output.narrow(1, 0, 4 * c.channels_in)
                    stego = iwt(output_stego)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)

                    #################
                    #   backward:   #
                    #################
                    output_stego = output_stego.cuda()
                    output_rev = torch.cat((output_stego, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)

                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    stego = stego.cpu().numpy().squeeze() * 255
                    np.clip(stego, 0, 255)
                    psnr_temp = calculate_psnr(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = calculate_psnr(cover, stego)
                    psnr_c.append(psnr_temp_c)

                logger.info('epoch: {}, training,  total_loss: {}, S_loss: {}, R_loss: {}, F_loss: {}'.format(i_epoch, np.array(loss_history).mean(), np.array(g_loss_history).mean(), np.array(r_loss_history).mean(), np.array(l_loss_history).mean()))
                logger.info('epoch: {}, testing, | stego_avg_psnr: {:.2f}, |secref_avg_psnr: {:.2f}'.format(i_epoch, np.mean(psnr_c), np.mean(psnr_s)))


        if i_epoch > 20 and (loss_history[-1][0] > 0.02 or math.isnan(loss_history[-1][0])):
                                                    
            r_epoch = i_epoch - (i_epoch % c.hinet_save_freq)
            r_epoch = (r_epoch-c.hinet_save_freq) if (i_epoch % c.hinet_save_freq) == 0 else r_epoch
            net.load_state_dict(torch.load(os.path.join(model_save_path, 'checkpoint_%.4i' % r_epoch + '.pt')))
            optim = torch.optim.Adam(net.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
            weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
            logger.info('epoch: {}, training breakdown, reload model'. format(i_epoch))
            
        if i_epoch > 0 and (i_epoch % c.hinet_save_freq) == 0:
            torch.save(net.state_dict(), os.path.join(model_save_path, 'checkpoint_%.4i' % i_epoch + '.pt'))
        weight_scheduler.step()



