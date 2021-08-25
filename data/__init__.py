import numpy as np
from torch.utils import data
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
from data.cityscapes_dataset_label_pseudo import cityscapesDataSetPseudoLabel
from data.synthia_dataset import SYNDataSet
from data.gta5_dataset_test  import GTA5DataSet_test
IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760)}
cs_size_test = {'cityscapes': (1344,576)}

def CreateSrcDataLoader(args):
    if args.source == 'gta5':
        source_dataset = GTA5DataSet( args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'], 
                                      resize=image_sizes['gta5'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    elif args.source == 'synthia':
        source_dataset = SYNDataSet( args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'],
                                      resize=image_sizes['synthia'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')
    
    source_dataloader = data.DataLoader( source_dataset, 
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         num_workers=args.num_workers, 
                                         pin_memory=True )    
    return source_dataloader



def CreateTestDataLoader(args):
    if args.set == 'val':
        test_dataset = GTA5DataSet_test( args.data_dir_test, args.data_list_test, crop_size=image_sizes['cityscapes'], 
                                      resize=image_sizes['gta5'] ,mean=IMG_MEAN)
    elif args.source == 'synthia':
        source_dataset = SYNDataSet( args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'],
                                      resize=image_sizes['synthia'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')
    
    test_dataloader = data.DataLoader( test_dataset, 
                                         batch_size=1,
                                         shuffle=False, 
                                         pin_memory=True )    
    return test_dataloader


def CreateTrgDataLoader(args):
    if args.set == 'train' or args.set == 'trainval':
        target_dataset = cityscapesDataSetLabel( args.data_dir_target, 
                                                 args.data_list_target, 
                                                 crop_size=image_sizes['cityscapes'], 
                                                 mean=IMG_MEAN, 
                                                 max_iters=args.num_steps * args.batch_size, 
                                                 set=args.set )
    else:
        target_dataset = cityscapesDataSet( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=cs_size_test['cityscapes'],
                                            mean=IMG_MEAN,
                                            set=args.set )

    if args.set == 'train' or args.set == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader


def CreateTrgDataLoader_trainset(args):
    
    target_dataset = cityscapesDataSet( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=cs_size_test['cityscapes'],
                                            mean=IMG_MEAN,
                                            set='train' )

    
    
    target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader


# use it to load pseudo labels for contrastive loss
def CreateTrgDataLoaderPseudo(args):
    if args.set == 'train' or args.set == 'trainval':
        target_dataset = cityscapesDataSetPseudoLabel( args.data_dir_target, 
                                                     args.label_folder,
                                                     args.data_list_target, 
                                                     crop_size=image_sizes['cityscapes'], 
                                                     mean=IMG_MEAN, 
                                                     max_iters=args.num_steps * args.batch_size, 
                                                     set='train' )
    else:
        target_dataset = cityscapesDataSet( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=cs_size_test['cityscapes'],
                                            mean=IMG_MEAN,
                                            set='train' )

    if args.set == 'train' or args.set == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader
