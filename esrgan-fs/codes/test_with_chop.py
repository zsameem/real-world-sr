import os.path as osp
import logging
import time
import argparse
import torch
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)
scale = opt['scale']
# divide image into divXdiv overlapping patches
divs = opt['divs']
util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
print('===========', opt['datasets'])
#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        # Algorithm:
        # Chop image into overlapping patches. Degree of overlap is determined
        # by the receptive field of the network.
        # Feed chopped up images to the network and reassmle the output back.
        t = data['LQ'] # Image data in batch x channel x height x width still on cpu
        img_path = data['LQ_path']
        [b, c, h, w] = t.size()
        print("Image size = {}, dividing into {}x{} patches for Inference".format(t.size(), divs, divs))

        sr_image = torch.zeros((b, c, scale*h, scale*w)).float()
        p_h = h // divs
        p_w = w // divs
        pad_h = 50
        pad_w = 50
        for i in range(divs):
            print('|', end='')
            
            if i == 0:
                sh = 0
                valid_sh = 0
                valid_psh_offset = 0
            else:
                sh = i*p_h - pad_h
                valid_sh = i*p_h
                valid_psh_offset = pad_h
            
            if i == divs - 1:
                eh = h
                valid_eh = h
                valid_peh_offset = 0
            else:
                eh = (i+1)* p_h + pad_h
                valid_eh = (i+1)*p_h
                valid_peh_offset = pad_h

            for j in range(divs):
                if j == 0:
                    sw = 0
                    valid_sw = 0
                    valid_psw_offset = 0
                else:
                    sw = j*p_w - pad_w
                    valid_sw = j*p_w
                    valid_psw_offset = pad_w

                if j == divs - 1:
                    ew = w
                    valid_ew = w
                    valid_pew_offset = 0
                else:
                    ew = (j+1)*p_w + pad_w
                    valid_ew = (j+1)*p_w
                    valid_pew_offset = pad_w

                # print("Slicing [{}:{}, {},{}], \t valid range [{}:{}, {}:{}]".format(
                #     sh, eh, sw, ew, valid_sh, valid_eh, valid_sw, valid_ew))
                
                with torch.no_grad():
                    lr = t[:,:, sh:eh, sw:ew]
                    data_dict = {}
                    data_dict['LQ'] = lr
                    # forward pass on the patch
                    out = model.netG(lr.cuda().half())
                    out = out.cpu()
                    [_,__, patch_h, patch_w] = out.size()
                    # Put sr patch in the correct position in the big image
                    sr_image[ : , :, 
                        valid_sh*scale:valid_eh*scale, 
                        valid_sw*scale:valid_ew*scale ] = out[:,:,
                        scale*valid_psh_offset:patch_h-scale*valid_peh_offset, 
                        scale*valid_psw_offset:patch_w-scale*valid_pew_offset ]
                    
                    print(u'\u2713 |', end='', flush=True)
                    # print(out.shape)
            print(' ')
                
        # slice tensor into divsxdivs parts 
        sr_img = util.tensor2img(sr_image)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        