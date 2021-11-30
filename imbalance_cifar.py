import torch
import torchvision
import torchvision.transforms as transforms
import copy
import numpy as np

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False,noise_type='none',noise_ratio=0.4):
        super(IMBALANCEMNIST, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        if(noise_type != 'none'):
            self.gen_misclassfied_data(type=noise_type,ratio=noise_ratio)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = torch.tensor(new_data)
        self.targets = torch.tensor(new_targets)

    def gen_misclassfied_data(self,type='sym',ratio='0.4'):
        if type == 'sym':
            new_labels = self.get_sym_noise_in_label(self.targets,ratio)
        elif type == 'asym':
            new_labels = self.get_rand_asym_noise_in_label(self.targets,ratio)
        elif type == 'custom':
            error_map = {0:0,1:7,2:2,3:3,4:4,5:5,6:6,7:1,8:3,9:9}
            new_labels = self.get_mapped_asym_noise_in_label(self.targets,error_map,ratio)
        self.original_target = copy.deepcopy(self.targets)
        self.targets = torch.tensor(new_labels)
        pass

    def get_sym_noise_in_label(labels, ratio):
        """The labels must be a numeric value, produces random symmetric noise"""
        new_labels = []
        num_cls = np.max(labels) + 1
        for i, label in enumerate(labels):
            if np.random.rand() < ratio:
                new_label = label
                while new_label == label:
                    new_label = np.random.randint(num_cls)
                new_labels.append(new_label)
            else:
                new_labels.append(label)
        return np.array(new_labels)
    
    def get_rand_asym_noise_in_label(labels,ratio):
        '''The labels must contain a numeric value, produces a random asymmetric noise'''
        total_labels = np.max(labels) + 1
        original_mappping = np.arange(total_labels)
        new_labels = []
        while True:
            new_mapping = np.random.permutation(total_labels)
            if np.any(original_mappping == new_mapping):
                continue
            else:
                break
        
        for i, label in enumerate(labels):
            if np.random.rand() < ratio:
                # This converts the label to new label depending upon the map we created
                new_label = new_mapping[label]
                new_labels.append(new_label)
            else:
                new_labels.append(label)

        return np.array(new_labels)

    def get_mapped_asym_noise_in_label(labels,new_mapping,ratio):
        '''The labels must contain a numeric value, mapping should be a dict {0:0,1:7,2:2,3:3,4:4,5:5,6:6,7:1,8:3,9:9} produces a random asymmetric noise'''
        new_labels = []
        for i, label in enumerate(labels):
            if np.random.rand() < ratio:
                # This converts the label to new label depending upon the map we created
                new_label = new_mapping[label]
                new_labels.append(new_label)
            else:
                new_labels.append(label)
        return np.array(new_labels)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()