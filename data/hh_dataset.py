import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import sys
sys.path.append('/home/akira/Project/simplestCNN_sparse2dense')
from Dataset.DatasetLoader.syncdataset import SyncDataset
from Dataset.DatasetLoader.ImageTensorBinLoader import ImageTensorBinLoader
import os

class hhDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc




        basic_dir = '/home/akira/Project/simplestCNN_sparse2dense/data/32-128data'
        bin32file = os.path.join(basic_dir, "Lidar32.bin")
        bin128file = os.path.join(basic_dir, "Lidar128.bin")
        bin32pa = os.path.join(basic_dir, "Lidar32.pa")
        bin128pa = os.path.join(basic_dir, "Lidar128.pa")
        rimg32loader = ImageTensorBinLoader(bin32file, bin32pa)
        rimg128loader = ImageTensorBinLoader(bin128file, bin128pa)
        self.rimgdataset = SyncDataset((0, 500))
        self.rimgdataset.registerDataLoader(rimg32loader, "rimg32")
        self.rimgdataset.registerDataLoader(rimg128loader, "rimg128")







    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))
        #
        # # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        #
        # A = A_transform(A)
        # B = B_transform(B)

        dic = self.rimgdataset[index]
        A = dic['rimg32']
        B = dic['rimg128']
        A = A.unsqueeze(0).float()
        B = B.unsqueeze(0).float()

        return {'A': A, 'B': B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.rimgdataset)
