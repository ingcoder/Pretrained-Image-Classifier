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

### Get Pet Labels
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
   
### Image Classifier
The classifier class was created by Udacity.com and used the resnet18, alexnet and vgg16 CNN models. 

    import ast
    from PIL import Image
    import torchvision.transforms as transforms
    from torch.autograd import Variable
    import torchvision.models as models
    from torch import __version__

    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

    # obtain ImageNet labels
    with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
        imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

    def classifier(img_path, model_name):
        # load the image
        img_pil = Image.open(img_path)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 

    # apply model to input
    model = models[model_name]

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()
    
    # apply data to model - adjusted based upon version to account for 
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)

    # pytorch versions less than 0.4
    else:
        # apply data to model
        output = model(data)

    # return index corresponding to predicted class
    pred_idx = output.data.numpy().argmax()

    return imagenet_classes_dict[pred_idx]

### Use Pre-trained Classifier
Define classify_images function. 
Creates classifier labels with classifier function, compares pet labels to 
the classifier labels, and adds the classifier label and the comparison of 
the labels to the results dictionary using the extend function.
This function doesn't return anything because the 
results_dic dictionary that is passed into the function is a mutable 
data type so no return is needed.

    from classifier import classifier 
    i = 0
    
    for filename in results_dic:      
        test_image = images_dir+filename
        image_classification = classifier(test_image, model)
        edit_image_classifier = image_classification.lower().strip()
        pet_label = results_dic[filename][0]
        if i < 40:
            if pet_label in edit_image_classifier:            
                results_dic[filename].extend([edit_image_classifier, 1])
            else: 
                results_dic[filename].extend([edit_image_classifier, 0])
            i += 1
        else: 
            break
    None 
 
### Print results
Prints summary results on the classification and then prints incorrectly 
classified dogs and incorrectly classified dog breed

    def print_results(results_dic, results_stats_dic, model, 
                  print_incorrect_dogs = False, print_incorrect_breed = False):
                  
    #print summary on classification
     message = "\n This is the result for CNN model: {}.\n number of images {}. % of correct dog images: {}.\n 
     number of dog images: {}. % of         correct breed: {}.\n number of not a dog image:{}. % of 
     correct not dog bread {}.".format(model.upper(), results_stats_dic.get('n_images'),   
     results_stats_dic.get('pct_correct_dogs'), results_stats_dic.get('n_dogs_img'), 
     results_stats_dic.get('pct_correct_breed'), results_stats_dic.get('n_notdogs_img'), 
     results_stats_dic.get('pct_correct_notdogs'))

     print(message)

    #conditional print misclassified dogs
    if (print_incorrect_dogs or print_incorrect_breeds) and (results_stats_dic['n_correct_dogs'] !=  
    results_stats_dic['n_correct_breed'] or (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_non_dogs'] != 
    results_stats_dic['n_images']) ):
        
        print('There were mismatches')
      
        for key in results_dic:
            #print misclassified dogs
            if sum(results_dic[key][3:]) == 1:
                print('Missclassification. One label classified as dog {0:100}'.format(results_dic[key][0]))
                
            
            elif sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0:
                print('Missclassification. One label classified breed {:>30}'.format(results_dic[key][1]))
          
    None
                

   
   
   
   
   
 

