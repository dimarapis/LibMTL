import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from aspp import DeepLabHead
#from create_dataset import NYUv2
from mine.create_dataset2 import warehouseSIM
from LibMTL import Trainer
from LibMTL.model import resnet_dilated
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
import wandb

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cpu')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=2, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=2, type=int, help='batch size for test')
    parser.add_argument('--resolution', default='mini', type=str, help='mini,std,full')#mini=320x180, std=534x300, full=640x360
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # prepare dataloaders
        # define dataset
        
    #print(params)
    wandb.init(project='libmtlnyu',entity='wandbdimar',
               name='{}_{}_{}_{}'.format(params.arch,params.weighting,params.scheduler,params.aug))
        #name='{}_{}_{}_{}_{}'.format(opt.data.dataset,opt.network.task,opt.network.weight,opt.network.archit,opt.network.grad_method),
        #config = wandb_config)
    force_cudnn_initialization()
    nyuv2_train_set = warehouseSIM(root=params.dataset_path, train=True, augmentation=True, params=params)
    nyuv2_test_set = warehouseSIM(root=params.dataset_path, train=False, params=params)
    
    #nyuv2_train_set = NYUv2(root=params.dataset_path, mode=params.train_mode, augmentation=params.aug)
    #nyuv2_test_set = NYUv2(root=params.dataset_path, mode='test', augmentation=False)
    
    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    
    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    
    # define tasks
    task_dict = {'segmentation': {'metrics':['mIoU', 'pixAcc'], 
                              'metrics_fn': SegMetric(),
                              'loss_fn': SegLoss(),
                              'weight': [1, 1]}, 
                 'depth': {'metrics':['abs_err', 'rel_err'], 
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]},
                 'normal': {'metrics':['mean', 'median', '<11.25', '<22.5', '<30'], 
                            'metrics_fn': NormalMetric(),
                            'loss_fn': NormalLoss(),
                            'weight': [0, 0, 1, 1, 1]}}
    
    # define encoder and decoders
    def encoder_class(): 
        return resnet_dilated('resnet50')
    num_out_channels = {'segmentation': 23, 'depth': 1, 'normal': 3}
    
    decoders = nn.ModuleDict({task: DeepLabHead(2048, 
                                                num_out_channels[task]) for task in list(task_dict.keys())})
    
    class NYUtrainer(Trainer):
        def __init__(self, wandb_run, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(NYUtrainer, self).__init__(wandb_run = wandb_run,
                                            task_dict=task_dict, 
                                            weighting=weighting_method.__dict__[weighting], 
                                            architecture=architecture_method.__dict__[architecture], 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)

        def process_preds(self, preds):
            sizes = {'mini': (180,320), 'std': (300,534), 'full': (360,640)}
            img_size = sizes[params.resolution]
            for task in self.task_name:
                print(preds[task].shape)
                preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
            return preds
    
    NYUmodel = NYUtrainer(wandb_run = wandb.run,
                          task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)
    NYUmodel.train(nyuv2_train_loader, nyuv2_test_loader, 200)
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)