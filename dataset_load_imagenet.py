from torchvision import datasets
import torchvision
import new_ImageFolder


# def get_training_augmentation():
#     train_transform = [
#      #albu.IAAAdditiveGaussianNoise(p=0.5),
#      #albu.IAAAdditiveGaussianNoise(p=0.5),
#      #albu.ShiftScaleRotate(p=0.5),
#      #albu.IAAPerspective(p=0.5),
#     albu.OneOf([
#         #albu.VerticalFlip(),
#         albu.HorizontalFlip(),
#     ], p=0.5),
#
#      albu.Rotate(p=0.5),
#      albu.RandomResizedCrop(32,32, scale=(0.7, 1.0),p=0.5),
#      #albu.RandomBrightnessContrast(p=0.3),
#      #albu.CLAHE(p=0.3),
#      #albu.Blur(p=0.3)
#      #albu.ChannelShuffle(p=0.5),
#      #albu.RandomGamma(p=0.5),
#      #albu.RandomGridShuffle(grid=(3, 3), p=0.5),
#      #albu.RandomRotate90(p=0.5),
#     ]
#     return albu.Compose(train_transform,p=0.5)
#
#
#
# class MyDataset(object):
#     def __init__(self, batch_size, use_gpu, num_workers, percentage_used, path):
#         trainloader=[]
#         train_labels=[]
#         testloader=[]
#         test_labels=[]
#         self.batch_size = batch_size
#         self.percentage_used = percentage_used
#         self.path=path
#
#         transform = [torchvision.transforms.RandomRotation(30), torchvision.transforms.RandomHorizontalFlip(),
#                      torchvision.transforms.RandomVerticalFlip(),
#                      #torchvision.transforms.ColorJitter(brightness=(0, 50), contrast=(0, 30), saturation=(0, 40),hue=(-0.5, 0.5)),
#                      torchvision.transforms.RandomResizedCrop(size=(28,28), scale=(0.9, 1.0))]
#
#         train_transforms = torchvision.transforms.Compose([
#             #torchvision.transforms.Resize(size=(32, 32)),
#             #torchvision.transforms.RandomApply(random_pick(transform), p=0.5),
#             torchvision.transforms.ToTensor(),
#         ])
#
#         test_transforms = torchvision.transforms.Compose([
#             #torchvision.transforms.Resize(size=(32, 32)),
#             torchvision.transforms.ToTensor(),
#         ])
#
#
#         train_dataset = datasets.ImageFolder(path+'/train', transform=train_transforms)
#         train_idx=[i for i in range(len(train_dataset.samples))]
#         train_loader = torch.utils.data.DataLoader(train_dataset, sampler=BalancedBatchSampler(train_dataset, train_idx), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True) #for each batch we have batch-size/class instances.
#
#         test_dataset = datasets.ImageFolder(path + '/test', transform=test_transforms)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
#
#         self.trainloader = train_loader
#         self.testloader = test_loader
#         self.num_classes = len(train_dataset.classes)
#         print("Full Dataset size: ", len(train_dataset.samples), " Dataset percentage used: ", percentage_used * 100, "% ","Classes: ", self.num_classes, "Test loader set: ", len(test_dataset.samples))
#
#
# class MyDataset_cross_val(object):
#     def __init__(self, batch_size, path, balanced_classes):
#         self.batch_size = batch_size
#         self.path=path
#
#         train_transforms = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((32,32)),
#             ToNumpy(),
#             AlbumentationToTorchvision(get_training_augmentation()),
#             torchvision.transforms.ToTensor()
#         ])
#
#         # test_transforms = torchvision.transforms.Compose([
#         #     torchvision.transforms.ToTensor(),
#         # ])
#
#
#
#
#         train_dataset = ImageFolder(path, transform=train_transforms)
#         num_train = len(train_dataset)
#         indices = list(range(num_train))
#         split = int(np.floor(0.2 * num_train))
#
#         np.random.seed(np.random.randint(0,1000))
#         np.random.shuffle(indices)
#         train_idx, valid_idx = indices[split:], indices[:split]
#         train_sampler = SubsetRandomSampler(train_idx)
#         test_sampler = SubsetRandomSampler(valid_idx)
#
#
#
#         train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                    sampler=Partial_BalancedBatchSampler(train_dataset, train_idx,
#                                                                                         balanced_classes=balanced_classes,
#                                                                                         num_classes=len(
#                                                                                             train_dataset.classes)),
#                                                    batch_size=batch_size, shuffle=False, num_workers=16,
#                                                    pin_memory=True)
#
#         test_loader = torch.utils.data.DataLoader(train_dataset, sampler=test_sampler, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
#
#
#
#         self.trainloader = train_loader
#         self.testloader = test_loader
#         self.num_classes = len(train_dataset.classes)
#         print("Full Dataset size: ", len(train_dataset), "Classes: ", self.num_classes, "Test loader set: ",len(test_loader)*batch_size)
#
#
# class eggsmentations(object):
#     def __init__(self, batch_size, path, balanced_classes):
#         self.batch_size = batch_size
#         self.path=path
#         self.balanced_classes=balanced_classes
#
#         normalize=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#
#         train_transforms = torchvision.transforms.Compose([
#             #torchvision.transforms.Resize((32, 32)),
#             ToNumpy(),
#             AlbumentationToTorchvision(get_training_augmentation()),
#             torchvision.transforms.ToTensor(),
#             #normalize
#         ])
#
#         test_transforms = torchvision.transforms.Compose([
#             #torchvision.transforms.Resize((32,32)),
#             torchvision.transforms.ToTensor(),
#             #normalize
#         ])
#
#
#         train_dataset = datasets.ImageFolder(self.path + '/train', transform=train_transforms)
#
#         train_idx = [i for i in range(len(train_dataset))]
#
#         train_loader = torch.utils.data.DataLoader(train_dataset,sampler=Partial_BalancedBatchSampler(train_dataset, train_idx, balanced_classes=balanced_classes, num_classes=len(train_dataset.classes)),batch_size=batch_size, shuffle=False, num_workers=16,pin_memory=True)
#         #train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=False, num_workers=16,pin_memory=True)
#
#         test_dataset = datasets.ImageFolder(self.path + '/test', transform=test_transforms)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
#
#         self.trainloader = train_loader
#         self.testloader = test_loader
#         self.num_classes = len(train_dataset.classes)
#         print("Full Dataset size: ", len(train_dataset),"Classes: ", self.num_classes, "Test loader set: ", len(test_dataset.samples))
#         # for data,label in train_loader:
#         #     class_instances = [data[(label == i)] for i in range(0, self.num_classes)]
#         #     print([len(i) for i in class_instances])
#         # exit()


class MyDataset_plus_aug(object):
    def __init__(self, path, channels, img_size, dataset_name):
        self.path = path
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.channels = channels
        #Imagenet preprocessing
        # train_transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomResizedCrop(224),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        #
        # test_transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(256),
        #     torchvision.transforms.CenterCrop(224),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        #
        # train_dataset = datasets.ImageFolder(self.path+'/train', transform=train_transforms)
        # test_dataset = datasets.ImageFolder(self.path+'/test', transform=test_transforms)

        #train_idx = [i for i in range(len(train_dataset))]

        # train_loader = torch.utils.data.DataLoader(train_dataset,sampler=Partial_BalancedBatchSampler(train_dataset, train_idx,
        #                                                                                 balanced_classes=balanced_classes,
        #                                                                                 num_classes=len(
        #                                                                                     train_dataset.classes)),
        #                                            batch_size=batch_size, shuffle=False, num_workers=8,
        #                                            pin_memory=True)

        if self.dataset_name == 'Imagenet-ILSVRC2012':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(self.img_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(self.img_size),
                #torchvision.transforms.CenterCrop(self.img_size),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])


        train_dataset = new_ImageFolder.ImageFolder(self.path, transform=train_transforms)

        test_dataset = new_ImageFolder.ImageFolder(self.path, transform=test_transforms)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        #b_balanced=BalanceBatchSampler(train_dataset.targets, p=len(train_dataset.classes), k=50)
        # b_balanced = torch.utils.data.distributed.DistributedSampler(b_balanced,shuffle=False)

        # train_loader = torch.utils.data.DataLoader(train_dataset, sampler=b_balanced, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        #
        #
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
        #                                           pin_memory=True)
