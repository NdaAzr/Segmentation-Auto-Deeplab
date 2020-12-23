# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:44:45 2020

@author: Neda
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:30:04 2020

@author: user
"""

from torch.utils.data import DataLoader
import torch.utils.data.distributed
from custom_dataset import CustomDataset
import torch
import glob
import numpy
#
#folder_data = glob.glob("/home/student1/Neda-project/data-seg-small/resize_imgae_Unet_1/*.png") # no augmnetation
#folder_mask = glob.glob("/home/student1/Neda-project/data-seg-small/resize_label_Unet_1/*.png")

folder_data = glob.glob("/home/student1/Neda-project/data-seg-small/resize_imgae_Unet_1/*.png") # no augmnetation
folder_mask = glob.glob("/home/student1/Neda-project/data-seg-small/resize_label_Unet_1/*.png")

len_data = len(folder_data)
print("count of dataset: ", len_data)


split_1 = int(0.6 * len(folder_data))
split_2 = int(0.8 * len(folder_data))

folder_data.sort()
folder_mask.sort()

train_image_paths = folder_data[:split_1]
print("count of train set  is: ", len(train_image_paths)) 
numpy.savetxt('im_training_path.csv', numpy.c_[train_image_paths], fmt=['%s'], comments='', delimiter = ",")   


train_image_paths1 = folder_data[:split_1//2]
print("count of train set 1 is: ", len(train_image_paths1)) 
numpy.savetxt('im_training_path_1.csv', numpy.c_[train_image_paths1], fmt=['%s'], comments='', delimiter = ",")   

train_image_paths2 = folder_data[split_1//2:split_1]
print("count of train set 2 is: ", len(train_image_paths2)) 
numpy.savetxt('im_training_path_2.csv', numpy.c_[train_image_paths2], fmt=['%s'], comments='', delimiter = ",")                    


valid_image_paths = folder_data[split_1:split_2]
print("count of validation image is: ", len(valid_image_paths))
numpy.savetxt('im_valid_path_1.csv', numpy.c_[valid_image_paths], fmt=['%s'], comments='', delimiter = ",")     


test_image_paths = folder_data[split_2:]
print("count of test images is: ", len(test_image_paths)) 
numpy.savetxt('im_testing_path_1.csv', numpy.c_[test_image_paths], fmt=['%s'], comments='', delimiter = ",")                    

train_mask_paths = folder_mask[:split_1]
train_mask_paths1 = folder_mask[:split_1//2]
train_mask_paths2 = folder_mask[split_1//2:split_1]
valid_mask_paths = folder_mask[split_1:split_2]
test_mask_paths = folder_mask[split_2:]
    
def make_data_loader(args, **kwargs):
    
    
    
    if args.dist:
        print("=> Using Distribued Sampler")
        
        if args.dataset == 'custom':
            
            if args.autodeeplab == 'search':
                
                num_class = 2

                train_set1 = CustomDataset(train_image_paths1, train_mask_paths1)
                sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1)
                train_loader1 = torch.utils.data.DataLoader(train_set1, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)
                print(len(train_loader1))
    
                train_set2 = CustomDataset(train_image_paths2, train_mask_paths2)
                sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2)
                train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size=args.batch_size, shuffle=False, sampler=sampler2, **kwargs)
                print(len(train_loader2))
                
                
                val_set = CustomDataset(valid_image_paths, valid_mask_paths)
                sampler3 = torch.utils.data.distributed.DistributedSampler(val_set)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=sampler3, **kwargs)
                
                test_set = CustomDataset(test_image_paths, test_mask_paths)
                sampler4 = torch.utils.data.distributed.DistributedSampler(val_set)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=sampler4, **kwargs)  
            
            elif args.autodeeplab == 'train':
                
                train_set = CustomDataset(train_image_paths, train_mask_paths)
                num_class = 2
                sampler_train = torch.utils.data.distributed.DistributedSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=sampler_train, **kwargs)

            else:
                
                raise Exception('autodeeplab param not set properly')                     

            if args.autodeeplab == 'search':
                return train_loader1, train_loader2, val_loader, test_loader, num_class
            elif args.autodeeplab == 'train':
                return train_loader, num_class, sampler1
        else:
            raise NotImplementedError                                                            
        
        
    

