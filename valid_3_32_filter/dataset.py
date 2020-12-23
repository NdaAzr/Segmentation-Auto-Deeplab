#from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader

from custom_dataset import CustomDataset
import torch
import glob
import numpy
import os
#
#folder_data = glob.glob("/home/student1/Neda-project/data-seg-small/resize_imgae_Unet_1/*.png") # no augmnetation
#folder_mask = glob.glob("/home/student1/Neda-project/data-seg-small/resize_label_Unet_1/*.png")
#
folder_data = glob.glob("E://Pytorch//U-net//my_data//resize_imgae_Unet_1\\*.png") # no augmnetation
folder_mask = glob.glob("E://Pytorch//U-net//my_data//resize_label_Unet_1\\*.png")

len_data = len(folder_data)
print("count of dataset: ", len_data)


split_1 = int(0.6 * len(folder_data))
split_2 = int(0.8 * len(folder_data))

folder_data.sort()
folder_mask.sort()

train_image_paths = folder_data[:split_1]
print("count of train images is: ", len(train_image_paths)) 
numpy.savetxt('im_training_path_1.csv', numpy.c_[train_image_paths], fmt=['%s'], comments='', delimiter = ",")                    


valid_image_paths = folder_data[split_1:split_2]
print("count of validation image is: ", len(valid_image_paths))
numpy.savetxt('im_valid_path_1.csv', numpy.c_[valid_image_paths], fmt=['%s'], comments='', delimiter = ",")     


test_image_paths = folder_data[split_2:]
print("count of test images is: ", len(test_image_paths)) 
numpy.savetxt('im_testing_path_1.csv', numpy.c_[test_image_paths], fmt=['%s'], comments='', delimiter = ",")                    


train_mask_paths = folder_mask[:split_1]
valid_mask_paths = folder_mask[split_1:split_2]
test_mask_paths = folder_mask[split_2:]

nclass = 2



def make_data_loader(args, **kwargs):
    
    if args.dataset == 'custom':
        
        num_class = 2
        train_dataset_1 = CustomDataset(train_image_paths, train_mask_paths)
        train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
        
#        train_dataset_2 = CustomDataset(train_image_paths_2, train_mask_paths_2)
#        train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
        
        valid_dataset = CustomDataset(valid_image_paths, valid_mask_paths)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
        
        test_dataset = CustomDataset(test_image_paths, test_mask_paths)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)  
    
    return train_loader_1, train_loader_1, val_loader, test_loader, num_class

#
#def make_data_loader(args, **kwargs):
#    if args.dist:
#        print("=> Using Distribued Sampler")
#        if args.dataset == 'cityscapes':
#            if args.autodeeplab == 'search':
#                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
#                num_class = train_set1.NUM_CLASSES
#                sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1)
#                sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2)
#                train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)
#                train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=False, sampler=sampler2, **kwargs)
#
#            elif args.autodeeplab == 'train':
#                train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
#                num_class = train_set.NUM_CLASSES
#                sampler1 = torch.utils.data.distributed.DistributedSampler(train_set)
#                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)
#
#            else:
#                raise Exception('autodeeplab param not set properly')
#
#            val_set = cityscapes.CityscapesSegmentation(args, split='val')
#            test_set = cityscapes.CityscapesSegmentation(args, split='test')
#            sampler3 = torch.utils.data.distributed.DistributedSampler(val_set)
#            sampler4 = torch.utils.data.distributed.DistributedSampler(test_set)
#            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=sampler3, **kwargs)
#            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=sampler4, **kwargs)
#
#            if args.autodeeplab == 'search':
#                return train_loader1, train_loader2, val_loader, test_loader, num_class
#            elif args.autodeeplab == 'train':
#                return train_loader, num_class, sampler1
#        else:
#            raise NotImplementedError
#
#    else:
#        if args.dataset == 'pascal':
#            train_set = pascal.VOCSegmentation(args, split='train')
#            val_set = pascal.VOCSegmentation(args, split='val')
#            if args.use_sbd:
#                sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
#                train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
#
#            num_class = train_set.NUM_CLASSES
#            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#            test_loader = None
#
#            return train_loader, train_loader, val_loader, test_loader, num_class
#
#        elif args.dataset == 'cityscapes':
#            if args.autodeeplab == 'search':
#                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
#                num_class = train_set1.NUM_CLASSES
#                train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
#                train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
#            elif args.autodeeplab == 'train':
#                train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
#                num_class = train_set.NUM_CLASSES
#                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#            else:
#                raise Exception('autodeeplab param not set properly')
#
#            val_set = cityscapes.CityscapesSegmentation(args, split='val')
#            test_set = cityscapes.CityscapesSegmentation(args, split='test')
#            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#
#            if args.autodeeplab == 'search':
#                return train_loader1, train_loader2, val_loader, test_loader, num_class
#            elif args.autodeeplab == 'train':
#                return train_loader, num_class
#
#
#
#        elif args.dataset == 'coco':
#            train_set = coco.COCOSegmentation(args, split='train')
#            val_set = coco.COCOSegmentation(args, split='val')
#            num_class = train_set.NUM_CLASSES
#            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#            test_loader = None
#            return train_loader, train_loader, val_loader, test_loader, num_class
#
#        elif args.dataset == 'kd':
#            train_set = kd.CityscapesSegmentation(args, split='train')
#            val_set = kd.CityscapesSegmentation(args, split='val')
#            test_set = kd.CityscapesSegmentation(args, split='test')
#            num_class = train_set.NUM_CLASSES
#            train_loader1 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#            train_loader2 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#
#            return train_loader1, train_loader2, val_loader, test_loader, num_class
#        else:
#            raise NotImplementedError
