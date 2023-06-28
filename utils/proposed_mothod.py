import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from collections import OrderedDict
import copy
import json
import random
from torch.optim.lr_scheduler import MultiStepLR 
import torch.optim as optim
from tqdm import tqdm


#####################################################################
#                    sparse mask generation                         #
#####################################################################
def generate_sparse_mask(model, sparse_ratio, sc, is_sampling=None, num_samples=None, train_loader_se=None, train_loader_st=None, device=None ,epoch=None):
    '''
    model: cover model.
    sparse_ratio: the number of '0' in numel(mask) 
    sc: selection criteria (random, large weight first, small weight first, large init_grad first, large global grad first)
    '''
    if sc == 'r': # random
        return random_based_sc(model, sparse_ratio)
    elif sc == 'lwf' or sc == 'swf': # large weight first or small weight first
        return weight_magnitude_based_sc(model, sparse_ratio, sc)
    else:  # large init_grad first or large global grad first
        return gradient_magnitude_based_sc(model, sparse_ratio, num_samples, is_sampling, sc, train_loader_se, train_loader_st, device ,epoch)

def random_based_sc(model, sr):
    num_w_in_m = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            num_w_in_m += torch.prod(torch.tensor(m.weight.shape)).item()
    n = int(num_w_in_m * sr)
    one_dim_mask = [0]*n + [1]*(num_w_in_m-n)
    random.shuffle(one_dim_mask)
    one_dim_mask = torch.from_numpy(np.array(one_dim_mask))
    
    sparse_masks = []
    start_idx = 0
    for m in model.modules():        
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            end_idx = start_idx + torch.prod(torch.tensor(m.weight.shape)).item()
            cur_layer_mask = one_dim_mask[start_idx:end_idx].view(m.weight.shape)
            start_idx = end_idx
            sparse_masks.append(cur_layer_mask)

    return sparse_masks

 
def weight_magnitude_based_sc(model, sr, sc):
    '''
    weight_magnitude_based_selection_criteria.
    '''
    compare = torch.gt if sc == 'lwf' else torch.lt
    sr = sr if sc == 'lwf' else (1-sr)
    
    weights_values = torch.Tensor([])
    for k, m in list(model.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            weights_values = torch.concat((weights_values, m.weight.data.abs().clone().view(-1)))

    n = int(len(weights_values)*sr)
    sorted_values, _ = torch.sort(weights_values, descending=True)
    # sorted_values, _ = torch.sort(weights_values)
    threshold = sorted_values[n-1]
    sparse_masks = []
    for k, m in list(model.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = compare(weight_copy, threshold).float()
            sparse_masks.append(mask)  
    return sparse_masks


def gradient_magnitude_based_sc(model, sr, num_samples, is_sampling, sc, train_loader_se, train_loader_st, device ,epoch):
    '''
    gradient_magnitude_based_selection_criteria.
    '''
    if sc == 'ligf':
        gradients_list_se = calculate_init_grad(model, train_loader_se, num_samples, device, is_sampling)
        gradients_list_st = calculate_init_grad(model, train_loader_st, num_samples, device, is_sampling)
    else:
        gradients_list_se = calculate_global_grad(model, train_loader_se, num_samples, device, is_sampling, epoch)
        gradients_list_st = calculate_global_grad(model, train_loader_st, num_samples, device, is_sampling, epoch)

    alpha = 1.0
    gradients_list = [x+alpha*y for x,y in zip(norm_(gradients_list_se), norm_(gradients_list_st))]          
    
    # to one dim gradients_list
    one_dim_grad_list = torch.tensor([]).to(device)
    for idx in range(len(gradients_list)):
        one_dim_grad_list = torch.cat((one_dim_grad_list, gradients_list[idx].view(-1)), dim=0)
    
    n = int(len(one_dim_grad_list)*sr)
    sorted_values, _ = torch.sort(one_dim_grad_list)
    threshold = sorted_values[n-1]

    sparse_masks = []
    for idx in range(len(gradients_list)):
        mask = torch.gt(gradients_list[idx], threshold).float()
        sparse_masks.append(mask)
    return sparse_masks


def calculate_init_grad(model, train_loader, num_samples, device, is_sampling):
    """
    Args:
        grad_type: (square or absolute) 
    """
    model = model.to(device)
    gradients_list = []

    # for name, param in model.named_parameters():
    #     gradients_dict[name] = torch.zeros_like(param).to(device)

    criterion = nn.CrossEntropyLoss()
    
    stream = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(stream):
        if is_sampling == True and batch_idx >= num_samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward() 

        idx = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if batch_idx == 0:
                    # gradients_list.append(m.weight.grad.data.clone())
                    gradients_list.append(m.weight.grad.data.clone().abs())
                else:
                    # gradients_list[idx] += m.weight.grad.data.clone()
                    gradients_list[idx] += m.weight.grad.data.clone().abs()
                idx += 1

        model.zero_grad()

        stream.set_description(
            " calculate init gradients "
        )

    return gradients_list


def calculate_global_grad(model, train_loader, num_samples, device, is_sampling, epoch):
    """
    Args:
        grad_type: (square or absolute) 
    """
    model = model.to(device)

    global_gradients_list = []
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            global_gradients_list.append(torch.zeros_like(m.weight).to(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-5, weight_decay=5e-6, amsgrad=False)
    scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    
    for ep in range(epoch): 
        stream = tqdm(train_loader)
        for batch_idx, (inputs, targets) in enumerate(stream):
            if is_sampling and batch_idx >= num_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward() 
            optimizer.step()

            idx = 0
            for m in model.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    # global_gradients_list[idx] += m.weight.grad.data.clone()
                    global_gradients_list[idx] += m.weight.grad.data.clone().abs()
                    idx += 1 
            
            stream.set_description(
            "calculate global gradients, Epoch: {epoch}".format(epoch=ep)
            )
            
    return global_gradients_list


def init_weights(model, random_seed=None):
    if random_seed != None:
        torch.manual_seed(random_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # m.weight.data *= 0.3
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# for m in self.modules():
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)

def norm_(grads_list):
    max_num = torch.max(grads_list[0])
    for idx in range(1, len(grads_list)):
        max_in_cur_layer = torch.max(grads_list[idx])
        max_num = max_num if max_in_cur_layer <= max_num else max_in_cur_layer
    
    for idx in range(len(grads_list)):
        # norm to [0, 1]
        grads_list[idx] = grads_list[idx]/max_num
    return grads_list


#####################################################################
#                         pluggable adapter                         #
#####################################################################
def reverse_mask(masks):
    # mask: a list contain several mask of conv or bn layer
    tmp = copy.deepcopy(masks)
    for idx in range(len(masks)):
        tmp[idx] = 1. - masks[idx]
    return tmp


def insert_adapter(model, sparse_mask, model_seed, is_sparse=True):
    '''
    model_seed: the model whose weight are initialized according to key(random seed).
    '''
    reverse_sparse_mask = reverse_mask(sparse_mask)
    idx_m = 0
    for [m, m_s] in zip(model.modules(), model_seed.modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if is_sparse == False:
                m.weight.data = m.weight.data.mul_(sparse_mask[idx_m])
            m.weight.data += m_s.weight.data.clone().mul_(reverse_sparse_mask[idx_m])
            idx_m += 1


def remove_adapter(model, sparse_mask):
    idx_m = 0
    for k, m in list(model.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.mul_(sparse_mask[idx_m])
            idx_m += 1


def insert_adapter_for_receiver(model, model_seed):
    idx_m = 0
    for [(k, m), (k_s, m_s)] in zip(model.named_modules(), model_seed.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            cur_reverse_mask  = m.weight.data.eq(0.).float()
            m.weight.data += m_s.weight.data.clone().mul_(cur_reverse_mask)
            idx_m += 1


def remove_adapter_for_receiver(model, model_seed):
    idx_m = 0
    for [(k, m), (k_s, m_s)] in zip(model.named_modules(), model_seed.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            equal_pos = (m.weight.data == m_s.weight.data).float()
            m.weight.data = m.weight.data.mul_(1. - equal_pos)
            idx_m += 1


#####################################################################
#                         batch normalization                       #
#####################################################################

def save_bn(model):
    r_ms = []; r_vs = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            r_ms.append(m.running_mean.clone())
            r_vs.append(m.running_var.clone())
    return r_ms, r_vs


def restore_bn(model, r_ms, r_vs):
    idx = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean = r_ms[idx]
            m.running_var = r_vs[idx]
            idx += 1


def record_bn_into_json(b_rm, b_rv, file_path):

    for idx in range(len(b_rm)):
        b_rm[idx] = b_rm[idx].tolist()
        b_rv[idx] = b_rv[idx].tolist()

    fs_se = {"b_rm_se": b_rm, "b_rv_se": b_rv}
    jsonString = json.dumps(fs_se)
    jsonFile = open(file_path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
 

def load_bn_from_json(file_path, device): 
    fileObject = open(file_path, "r")
    jsonContent = fileObject.read()
    fs_se = json.loads(jsonContent)
    b_rm_se = fs_se['b_rm_se']; b_rv_se = fs_se['b_rv_se']

    for idx in range(len(b_rm_se)):
        b_rm_se[idx] = torch.from_numpy(np.array(b_rm_se[idx])).float().to(device)
        b_rv_se[idx] = torch.from_numpy(np.array(b_rv_se[idx])).float().to(device)
    
    return b_rm_se, b_rv_se


def dwt_init(x):
    
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


def downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    if 'cuda' in x.type():
        down_features = torch.cuda.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    
    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features

def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout)).type(x.type())
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature