# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:15:51 2020

@author: Neda
"""

from custom_dataset import CustomDataset
import torch
import glob
import numpy


#
#folder_data = glob.glob("E://Pytorch//U-net//my_data//resize_imgae_Unet_1\\*.png") # no augmnetation
#folder_mask = glob.glob("E://Pytorch//U-net//my_data//resize_label_Unet_1\\*.png")


def make_data_loader(args, **kwargs):
    
    folder_data = glob.glob("/home/student1/Neda-project/data-seg-small/resize_imgae_Unet_1/*.png") # no augmnetation
    folder_mask = glob.glob("/home/student1/Neda-project/data-seg-small/resize_label_Unet_1/*.png")
    
    len_data = len(folder_data)
    print("count of dataset: ", len_data)
    
    
    split_1 = int(0.8 * len(folder_data))
    split_2 = int(0.2 * len(folder_data))
    
    folder_data.sort()
    folder_mask.sort()
    
    train_image_paths = folder_data[:split_1]
    print("count of train images is: ", len(train_image_paths)) 
    numpy.savetxt('im_training_path_1.csv', numpy.c_[train_image_paths], fmt=['%s'], comments='', delimiter = ",")                    
    
    
#    valid_image_paths = folder_data[split_1:split_2]
#    print("count of validation image is: ", len(valid_image_paths))
#    numpy.savetxt('im_valid_path_1.csv', numpy.c_[valid_image_paths], fmt=['%s'], comments='', delimiter = ",")     
    
    
    test_image_paths = folder_data[split_1:]
    print("count of test images is: ", len(test_image_paths)) 
    numpy.savetxt('im_testing_path_1.csv', numpy.c_[test_image_paths], fmt=['%s'], comments='', delimiter = ",")                    
    
    
    train_mask_paths = folder_mask[:split_1]
#    valid_mask_paths = folder_mask[split_1:split_2]
    test_mask_paths = folder_mask[split_1:]

    
    if args.dataset == 'custom':
        
        num_class = 2
        train_dataset = CustomDataset(train_image_paths, train_mask_paths)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=14, shuffle=True, num_workers=4, drop_last=True)
        

#        valid_dataset = CustomDataset(valid_image_paths, valid_mask_paths)
#        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
        
        test_dataset = CustomDataset(test_image_paths, test_mask_paths)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)  
    
    return train_loader, test_loader, num_class

