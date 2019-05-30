# Pretrained-Image-Classifier
!/usr/bin/env python3
 -*- coding: utf-8 -*-
                                                                             
 PROGRAMMER: Ingrid Barbosa
 DATE CREATED: 11/12/2018 
 This project was done as part of the Udacity pone degree program "AI Programming with Python"
 
 ## Description
This is a script to identify images of dog breeds using a pretrained classifier 
that was trained on 1.2 million images from ImageNet. 
The classifier uses three different deep learning models (convolutional neural network, short CNN):
- AlexNet 
- VGG16  
- ResNET18
The best model for the application will be determined by measuring performance time and prediction accuracy. 

## Code samples
Creates a dictionary of pet labels (results_dic) based upon the filenames 
of the image files. These pet image labels are used to check the accuracy 
of the labels that are returned by the classifier function, since the 
filenames of the images contain the true identity of the pet in the image.
    
    from os import listdir
    import re    
    
    def get_pet_labels(image_dir):    
    
    filename_list = listdir(image_dir)
    pet_labels = []
    results_dic = dict()
    
    #Create list with pet_labels
    #for i in range(len(filename_list)):
    for filename in filename_list: 
        if filename[0] != ".":
            filename = filename.rsplit('.jpg')
            name = re.sub(r'[0-9]+','', filename[0].lower().replace('_',' '))
            name = name.strip()
            #print(name, name.isalpha())
            pet_labels.append(name) 
        
    for idx in range(len(filename_list)):
        if filename_list[idx] not in results_dic:
            results_dic[filename_list[idx]] = [pet_labels[idx]]
            print("{:2d} file: {:>25s}".format(idx + 1, filename_list[idx]) )
        else: 
            print('Name {} already exist'.format(filename_list[idx]))
  
    #for key, value in results_dic.items():
        #print('Key:{} Value:{}'.format(key, value))
        
    return results_dic
 

