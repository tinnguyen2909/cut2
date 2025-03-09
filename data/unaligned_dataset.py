import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import re


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
        if not self.opt.serial_batches:
            # Shuffle the A_paths list
            random.shuffle(self.A_paths)
        
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
        
        # Apply random scaling if enabled and path matches pattern
        if hasattr(self.opt, 'enable_random_scale') and self.opt.enable_random_scale and random.random() < self.opt.enable_random_scale_prob:
            # Check if A image path matches the pattern
            if hasattr(self.opt, 'enable_scale_on') and re.search(self.opt.enable_scale_on, A_path):
                # Random scale factor between 0.33 and 0.9
                scale_factor = random.uniform(0.33, 0.9)
                # Get original size
                orig_width, orig_height = A_img.size
                # Calculate new size
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)
                # Resize image
                scaled_img = A_img.resize((new_width, new_height), Image.LANCZOS)
                # Create white background
                white_bg = Image.new('RGB', (orig_width, orig_height), (255, 255, 255))
                # Calculate position to center the scaled image
                paste_x = (orig_width - new_width) // 2
                paste_y = (orig_height - new_height) // 2
                # Paste scaled image onto white background
                white_bg.paste(scaled_img, (paste_x, paste_y))
                A_img = white_bg
                
            # Check if B image path matches the pattern
            if hasattr(self.opt, 'enable_scale_on') and re.search(self.opt.enable_scale_on, B_path):
                # Random scale factor between 0.33 and 0.9
                scale_factor = random.uniform(0.33, 0.6)
                # Get original size
                orig_width, orig_height = B_img.size
                # Calculate new size
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)
                # Resize image
                scaled_img = B_img.resize((new_width, new_height), Image.LANCZOS)
                # Create white background
                white_bg = Image.new('RGB', (orig_width, orig_height), (255, 255, 255))
                # Calculate position to center the scaled image
                paste_x = (orig_width - new_width) // 2
                paste_y = (orig_height - new_height) // 2
                # Paste scaled image onto white background
                white_bg.paste(scaled_img, (paste_x, paste_y))
                B_img = white_bg
        
        # Apply random rotation if enabled
        if hasattr(self.opt, 'enable_rotation') and self.opt.enable_rotation and random.random() < self.opt.enable_rotation_prob:
            # Check if A image path matches the pattern
            if hasattr(self.opt, 'enable_rotation_on') and re.search(self.opt.enable_scale_on, A_path):
                # Random rotation angle between -60 and +60 degrees
                rotation_angle = random.uniform(-60, 60)
                # Rotate image with white background fill
                A_img = A_img.rotate(rotation_angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
            
            # Check if B image path matches the pattern
            if hasattr(self.opt, 'enable_rotation_on') and re.search(self.opt.enable_scale_on, B_path):
                # Random rotation angle between -60 and +60 degrees
                rotation_angle = random.uniform(-60, 60)
                # Rotate image with white background fill
                B_img = B_img.rotate(rotation_angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
        
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
