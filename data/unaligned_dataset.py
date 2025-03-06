import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # Parse comma-separated directory strings
        self.dir_A_list = [dir.strip() for dir in opt.path_A.split(',')]
        self.dir_B_list = [dir.strip() for dir in opt.path_B.split(',')]
        
        # Load images from all directories for domain A
        self.A_paths = []
        for dir_A in self.dir_A_list:
            if os.path.exists(dir_A):
                paths = sorted(make_dataset(dir_A, opt.max_dataset_size - len(self.A_paths)))
                self.A_paths.extend(paths)
                if len(self.A_paths) >= opt.max_dataset_size:
                    break
        
        # Load images from all directories for domain B
        self.B_paths = []
        for dir_B in self.dir_B_list:
            if os.path.exists(dir_B):
                paths = sorted(make_dataset(dir_B, opt.max_dataset_size - len(self.B_paths)))
                self.B_paths.extend(paths)
                if len(self.B_paths) >= opt.max_dataset_size:
                    break
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)
        
        # Handle alpha channel by compositing against white background
        if A_img.mode in ('RGBA', 'LA') or (A_img.mode == 'P' and 'transparency' in A_img.info):
            # Create a white background image
            white_bg = Image.new('RGBA', A_img.size, (255, 255, 255, 255))
            # Paste the image on the background
            white_bg.paste(A_img, (0, 0), A_img)
            A_img = white_bg
            
        if B_img.mode in ('RGBA', 'LA') or (B_img.mode == 'P' and 'transparency' in B_img.info):
            # Create a white background image
            white_bg = Image.new('RGBA', B_img.size, (255, 255, 255, 255))
            # Paste the image on the background
            white_bg.paste(B_img, (0, 0), B_img)
            B_img = white_bg
            
        # Convert to RGB
        A_img = A_img.convert('RGB')
        B_img = B_img.convert('RGB')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
