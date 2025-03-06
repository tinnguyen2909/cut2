import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch.nn.functional as F


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        # Add new options for eye color and skin tone preservation
        parser.add_argument('--lambda_eye', type=float, default=5.0, help='weight for eye color preservation loss')
        parser.add_argument('--lambda_skin', type=float, default=1.0, help='weight for skin tone preservation loss')
        parser.add_argument('--use_face_parser', type=util.str2bool, default=True, help='use face parsing model for precise feature extraction')
        parser.add_argument('--lambda_segmentation', type=float, default=5.0, help='weight for segmentation consistency loss')

        # Add new options for edge preservation and color consistency
        parser.add_argument('--lambda_edge', type=float, default=5.0, help='weight for edge preservation loss')
        parser.add_argument('--lambda_color_consistency', type=float, default=3.0, help='weight for color consistency loss to prevent bleeding')
        parser.add_argument('--edge_threshold', type=float, default=0.05, help='threshold for detecting important edges')

        # Add option for ethnicity preservation
        parser.add_argument('--lambda_ethnicity', type=float, default=2.0, help='weight for ethnicity preservation loss')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        # Add new losses for eye color and skin tone preservation
        if opt.lambda_eye > 0.0:
            self.loss_names.append('eye')
        if opt.lambda_skin > 0.0:
            self.loss_names.append('skin')
        # Add segmentation loss
        if opt.lambda_segmentation > 0.0:
            self.loss_names.append('segmentation')

        # Add new losses for edge preservation and color consistency
        if opt.lambda_edge > 0.0:
            self.loss_names.append('edge')
        if opt.lambda_color_consistency > 0.0:
            self.loss_names.append('color')
        # Add ethnicity preservation loss
        if opt.lambda_ethnicity > 0.0:
            self.loss_names.append('ethnicity')

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            # Initialize face parser if needed
            if (opt.lambda_eye > 0.0 or opt.lambda_skin > 0.0) and opt.use_face_parser:
                try:
                    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
                    self.has_face_parser = True
                    self.face_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
                    self.face_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
                    self.face_model.to(self.device)
                    print("Face parsing model loaded successfully")
                    
                    # Define label indices from the face parser
                    self.skin_label = 1
                    self.left_eye_label = 4
                    self.right_eye_label = 5
                except ImportError:
                    print("Warning: transformers package not found. Using simple region-based extraction instead.")
                    self.has_face_parser = False
            else:
                self.has_face_parser = False

            # Register Sobel filters for edge detection
            if opt.lambda_edge > 0.0:
                # Define Sobel filters - note the shape change
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
                self.sobel_x = sobel_x.to(self.device)
                self.sobel_y = sobel_y.to(self.device)

            # Initialize race prediction model
            self.race_model = None
            self.race_model_loaded = False

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_face_parsing_masks(self, img):
        """Extract face parsing masks using the segformer model with caching"""
        # Create a cache key based on the image tensor data
        cache_key = hash(img.cpu().detach().numpy().tobytes())
        
        # If we already processed this exact image, return cached result
        if hasattr(self, 'face_parsing_cache') and cache_key in self.face_parsing_cache:
            return self.face_parsing_cache[cache_key]
        
        # If not in cache, process the image
        batch_size = img.size(0)
        img_np = img.detach().cpu().numpy().transpose(0, 2, 3, 1)  # BCHW -> BHWC
        img_np = (img_np + 1) * 127.5  # [-1,1] -> [0,255]
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Process each image in the batch and get segmentation masks
        all_masks = []
        
        for i in range(batch_size):
            # Convert to PIL Image
            from PIL import Image
            pil_img = Image.fromarray(img_np[i])
            
            # Process through face parser
            inputs = self.face_processor(images=pil_img, return_tensors="pt").to(self.device)
            outputs = self.face_model(**inputs)
            logits = outputs.logits
            
            # Resize output to match input image dimensions
            upsampled_logits = F.interpolate(
                logits,
                size=(img.size(2), img.size(3)),  # H x W
                mode='bilinear',
                align_corners=False
            )
            
            # Get label masks
            masks = upsampled_logits.argmax(dim=1)  # Shape: [1, H, W]
            all_masks.append(masks)
            
        # Combine all masks into a batch
        batch_masks = torch.cat(all_masks, dim=0)  # Shape: [B, H, W]
        
        # Cache the result
        if hasattr(self, 'face_parsing_cache'):
            self.face_parsing_cache[cache_key] = batch_masks
        
        return batch_masks

    def extract_eye_regions(self, img):
        """Extract eye regions from an image using face parsing or simple heuristic"""
        if self.has_face_parser:
            # Get face parsing masks
            masks = self.get_face_parsing_masks(img)
            
            # Create eye mask (combine left and right eyes)
            eye_mask = ((masks == self.left_eye_label) | (masks == self.right_eye_label)).float().unsqueeze(1)
            
            # Apply mask to the original image to get only eye regions
            eye_regions = img * eye_mask
            
            # Return the masked image
            return eye_regions
        else:
            # Fallback to simple heuristic
            b, c, h, w = img.size()
            eye_region = img[:, :, h//5:h//2, w//4:3*w//4]
            return F.interpolate(eye_region, size=(64, 128), mode='bilinear')

    def extract_skin_regions(self, img):
        """Extract skin regions from an image using face parsing or simple heuristic"""
        if self.has_face_parser:
            # Get face parsing masks
            masks = self.get_face_parsing_masks(img)
            
            # Create skin mask
            skin_mask = (masks == self.skin_label).float().unsqueeze(1)
            
            # Apply mask to get skin regions
            skin_regions = img * skin_mask
            
            return skin_regions
        else:
            # Fallback to simple heuristic
            b, c, h, w = img.size()
            return img[:, :, h//4:3*h//4, w//4:3*w//4]

    def compute_eye_color_loss(self, src, tgt):
        """Calculate loss for preserving eye color"""
        src_eyes = self.extract_eye_regions(src)
        tgt_eyes = self.extract_eye_regions(tgt)
        
        # For the face parser method, we need to handle potential empty masks
        if self.has_face_parser:
            # Create binary mask of non-zero pixels (where eyes were detected)
            src_mask = (torch.sum(src_eyes, dim=1, keepdim=True) != 0).float()
            tgt_mask = (torch.sum(tgt_eyes, dim=1, keepdim=True) != 0).float()
            
            # Combine masks - we only consider pixels where both source and target detected eyes
            combined_mask = src_mask * tgt_mask
            
            # If no eye pixels are detected, return zero loss
            if combined_mask.sum() < 1.0:
                return torch.tensor(0.0, device=self.device)
            
            # Apply combined mask to both images
            src_eyes_masked = src_eyes * combined_mask
            tgt_eyes_masked = tgt_eyes * combined_mask
            
            # Calculate mean color of eye regions for more stable comparison
            src_mean = torch.sum(src_eyes_masked, dim=[2, 3]) / (combined_mask.sum() + 1e-6)
            tgt_mean = torch.sum(tgt_eyes_masked, dim=[2, 3]) / (combined_mask.sum() + 1e-6)
            
            # Compare overall color distributions and local details
            color_loss = F.l1_loss(src_mean, tgt_mean)
            detail_loss = F.l1_loss(src_eyes_masked, tgt_eyes_masked) 
            
            return color_loss * 0.7 + detail_loss * 0.3
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_skin_tone_loss(self, src, tgt):
        """Calculate loss for preserving skin tone"""
        src_skin = self.extract_skin_regions(src)
        tgt_skin = self.extract_skin_regions(tgt)
        
        if self.has_face_parser:
            # Create binary mask of non-zero pixels (where skin was detected)
            src_mask = (torch.sum(src_skin, dim=1, keepdim=True) != 0).float()
            tgt_mask = (torch.sum(tgt_skin, dim=1, keepdim=True) != 0).float()
            
            # Combine masks
            combined_mask = src_mask * tgt_mask
            
            # If no skin pixels are detected, return zero loss
            if combined_mask.sum() < 1.0:
                return torch.tensor(0.0, device=self.device)
            
            # Calculate color statistics on masked regions
            src_mean = torch.sum(src_skin * combined_mask, dim=[2, 3]) / torch.sum(combined_mask, dim=[2, 3])
            tgt_mean = torch.sum(tgt_skin * combined_mask, dim=[2, 3]) / torch.sum(combined_mask, dim=[2, 3])
            
            # Compare color distributions
            return F.l1_loss(src_mean, tgt_mean)
        else:
            # Simple color statistics for heuristic method
            src_mean = torch.mean(src_skin, dim=[2, 3])  # Average over spatial dimensions
            tgt_mean = torch.mean(tgt_skin, dim=[2, 3])
            
            return F.l1_loss(src_mean, tgt_mean)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        # Clear the face parsing cache at the beginning of each forward pass
        if hasattr(self, 'has_face_parser') and self.has_face_parser:
            self.face_parsing_cache = {}

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def detect_edges(self, img):
        """Detect edges using Sobel filters"""
        # Convert to grayscale for edge detection
        if img.size(1) == 3:  # If image has 3 channels (RGB)
            # Convert to grayscale: 0.299 * R + 0.587 * G + 0.114 * B
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
        
        # Pad the input to maintain size after convolution
        padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        
        # Apply Sobel filters
        batch_size = gray.size(0)
        
        # Process each channel separately
        grad_x = F.conv2d(padded, weight=self.sobel_x, groups=1)
        grad_y = F.conv2d(padded, weight=self.sobel_y, groups=1)
        
        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Normalize
        grad_magnitude = grad_magnitude / grad_magnitude.max()
        
        return grad_magnitude, grad_x, grad_y

    def compute_edge_preservation_loss(self, source, target):
        """Compute edge preservation loss to maintain important features"""
        # Detect edges in source and target images
        source_edges, source_grad_x, source_grad_y = self.detect_edges(source)
        target_edges, target_grad_x, target_grad_y = self.detect_edges(target)
        
        # Identify important edges in source image (using threshold)
        important_edges = (source_edges > self.opt.edge_threshold).float()
        
        # Compute weighted edge preservation loss
        # This focuses on preserving important edges from source image
        edge_loss = F.l1_loss(target_edges * important_edges, source_edges * important_edges)
        
        # Compute gradient direction consistency loss for important edges
        # This helps maintain the correct edge orientation
        eps = 1e-6  # Small value to prevent division by zero
        source_grad_norm = torch.sqrt(source_grad_x**2 + source_grad_y**2 + eps)
        target_grad_norm = torch.sqrt(target_grad_x**2 + target_grad_y**2 + eps)
        
        source_grad_x_norm = source_grad_x / source_grad_norm
        source_grad_y_norm = source_grad_y / source_grad_norm
        target_grad_x_norm = target_grad_x / target_grad_norm
        target_grad_y_norm = target_grad_y / target_grad_norm
        
        # Compute cosine similarity between gradient directions
        direction_similarity = (source_grad_x_norm * target_grad_x_norm + 
                              source_grad_y_norm * target_grad_y_norm)
        
        # Convert similarity to a loss (1 - similarity)
        direction_loss = torch.mean((1.0 - direction_similarity) * important_edges)
        
        # Combine losses
        return edge_loss * 0.7 + direction_loss * 0.3

    def compute_color_consistency_loss(self, source, target):
        """Compute color consistency loss to prevent color bleeding"""
        # Detect edges to identify boundaries
        edges, _, _ = self.detect_edges(source)
        edges = (edges > self.opt.edge_threshold).float()
        
        # Dilate edges to create boundary regions
        kernel_size = 3
        padding = kernel_size // 2
        edge_regions = F.max_pool2d(edges, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Non-edge regions (where color consistency is enforced more strictly)
        non_edge_regions = 1.0 - edge_regions
        
        # Get color statistics in non-edge regions (mean and variance)
        # For each channel separately
        color_loss = 0.0
        
        # Process regions using local windows
        window_size = 7
        padding = window_size // 2
        
        # For efficiency, sample random positions rather than processing every pixel
        batch_size, channels, height, width = source.size()
        num_samples = 1000
        
        for _ in range(num_samples):
            # Select random position
            y = torch.randint(padding, height - padding, (1,)).item()
            x = torch.randint(padding, width - padding, (1,)).item()
            
            # Extract local patches
            source_patch = source[:, :, y-padding:y+padding+1, x-padding:x+padding+1]
            target_patch = target[:, :, y-padding:y+padding+1, x-padding:x+padding+1]
            region_mask = non_edge_regions[:, :, y-padding:y+padding+1, x-padding:x+padding+1]
            
            # If the patch is mostly in non-edge region, enforce color consistency
            if region_mask.mean() > 0.7:  # If more than 70% of patch is non-edge
                # Calculate color statistics
                source_mean = torch.mean(source_patch, dim=[2, 3])
                target_mean = torch.mean(target_patch, dim=[2, 3])
                
                source_std = torch.std(source_patch, dim=[2, 3])
                target_std = torch.std(target_patch, dim=[2, 3])
                
                # Color consistency loss
                color_loss += F.l1_loss(source_mean, target_mean) + 0.5 * F.l1_loss(source_std, target_std)
        
        # Normalize by number of samples
        color_loss = color_loss / num_samples
        
        # Check for abrupt color transitions (color bleeding)
        # Look at adjacent pixels in non-edge regions
        source_dx = torch.abs(source[:, :, :, 1:] - source[:, :, :, :-1]) * non_edge_regions[:, :, :, :-1]
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1]) * non_edge_regions[:, :, :, :-1]
        
        source_dy = torch.abs(source[:, :, 1:, :] - source[:, :, :-1, :]) * non_edge_regions[:, :, :-1, :]
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]) * non_edge_regions[:, :, :-1, :]
        
        # Penalize when target has larger color transitions than source (indicates bleeding)
        bleeding_loss_x = torch.mean(F.relu(target_dx - source_dx - 0.01))  # Small threshold for tolerance
        bleeding_loss_y = torch.mean(F.relu(target_dy - source_dy - 0.01))
        bleeding_loss = bleeding_loss_x + bleeding_loss_y
        
        return color_loss * 0.5 + bleeding_loss * 0.5

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
            
        # Calculate eye color preservation loss
        if self.opt.lambda_eye > 0.0:
            self.loss_eye = self.compute_eye_color_loss(self.real_A, self.fake_B) * self.opt.lambda_eye
        else:
            self.loss_eye = 0.0
            
        # Calculate skin tone preservation loss
        if self.opt.lambda_skin > 0.0:
            self.loss_skin = self.compute_skin_tone_loss(self.real_A, self.fake_B) * self.opt.lambda_skin
        else:
            self.loss_skin = 0.0
            
        # Calculate segmentation consistency loss
        if self.opt.lambda_segmentation > 0.0:
            self.loss_segmentation = self.compute_segmentation_loss(self.real_A, self.fake_B) * self.opt.lambda_segmentation
        else:
            self.loss_segmentation = 0.0

        # Calculate edge preservation loss
        if self.opt.lambda_edge > 0.0:
            self.loss_edge = self.compute_edge_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_edge
        else:
            self.loss_edge = 0.0
        
        # Calculate color consistency loss
        if self.opt.lambda_color_consistency > 0.0:
            self.loss_color = self.compute_color_consistency_loss(self.real_A, self.fake_B) * self.opt.lambda_color_consistency
        else:
            self.loss_color = 0.0
            
        # Calculate ethnicity preservation loss
        if self.opt.lambda_ethnicity > 0.0:
            self.loss_ethnicity = self.compute_ethnicity_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_ethnicity
        else:
            self.loss_ethnicity = 0.0

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_eye + self.loss_skin + \
                     self.loss_segmentation + self.loss_edge + self.loss_color + self.loss_ethnicity
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def compute_segmentation_loss(self, src, tgt):
        """Calculate loss to ensure consistent segmentation masks between source and target"""
        if not self.has_face_parser:
            return torch.tensor(0.0, device=self.device)
            
        # Get segmentation masks for source and generated images
        src_masks = self.get_face_parsing_masks(src)
        tgt_masks = self.get_face_parsing_masks(tgt)
        
        # Calculate pixel accuracy (percent of pixels with matching labels)
        mask_equals = (src_masks == tgt_masks).float()
        pixel_acc = mask_equals.mean()
        
        # Convert to one-hot for class-wise calculations
        batch_size, height, width = src_masks.size()
        num_classes = 19  # Number of classes in face-parsing model
        
        src_one_hot = F.one_hot(src_masks, num_classes).permute(0, 3, 1, 2).float()
        tgt_one_hot = F.one_hot(tgt_masks, num_classes).permute(0, 3, 1, 2).float()
        
        # Check which classes are present in each image
        src_classes_present = (torch.sum(src_one_hot, dim=[2, 3]) > 0) # Shape: [batch_size, num_classes]
        tgt_classes_present = (torch.sum(tgt_one_hot, dim=[2, 3]) > 0)
        
        # Identify classes present in source but missing in target
        # This directly penalizes when target masks don't have features from source masks
        missing_classes = src_classes_present & ~tgt_classes_present
        
        # Define importance weights for different facial features
        class_weights = torch.ones(num_classes, device=self.device)
        # Increase weights for important facial features
        class_weights[2] = 2.0    # nose
        class_weights[5] = 2.0    # Right eye
        class_weights[4] = 2.0   # left eye
        class_weights[10] = 2.0   # Mouth
        class_weights[3] = 2.0    # Glasses (eye_g)
        class_weights[15] = 2.0   # Earrings
        class_weights[16] = 2.0   # Earrings
        
        # Calculate missing feature penalty with importance weighting
        # Sum across the classes dimension, then average across batch
        missing_feature_penalty = ((missing_classes.float() * class_weights).sum(dim=1) / 
                                  (src_classes_present.float() * class_weights).sum(dim=1).clamp(min=1e-6)).mean()
        
        # Calculate IoU for each class (for classes present in source)
        intersection = torch.sum(src_one_hot * tgt_one_hot, dim=[2, 3])
        union = torch.sum(src_one_hot, dim=[2, 3]) + torch.sum(tgt_one_hot, dim=[2, 3]) - intersection
        
        # Calculate IoU only for classes that are present in source
        iou = torch.zeros_like(intersection)
        iou[src_classes_present] = intersection[src_classes_present] / (union[src_classes_present] + 1e-6)
        weighted_iou = (iou * class_weights).sum(dim=1) / (src_classes_present.float() * class_weights).sum(dim=1).clamp(min=1e-6)
        mean_weighted_iou = weighted_iou.mean()
        
        # Combined loss: pixel accuracy, IoU, and missing feature penalty
        # Higher weight on missing feature penalty to emphasize feature preservation
        final_loss = 0.2 * (1.0 - pixel_acc) + 0.3 * (1.0 - mean_weighted_iou) + 0.5 * missing_feature_penalty
        
        return final_loss

    def compute_ethnicity_preservation_loss(self, source, target):
        """Compute ethnicity preservation loss to keep the same race/ethnicity between source and target
        
        Args:
            source (tensor): Source image tensor
            target (tensor): Target image tensor
            
        Returns:
            tensor: Ethnicity preservation loss
        """
        import sys
        import os
        
        # Ensure fairface is in the path
        fairface_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fairface')
        if fairface_dir not in sys.path:
            sys.path.append(fairface_dir)
            
        try:
            from fairface.predict_race import load_model
            import torch
            import torch.nn.functional as F
            import torchvision.transforms as transforms
        except ImportError:
            print("Error importing FairFace modules. Make sure the fairface directory is properly set up.")
            return torch.tensor(0.0, device=source.device)
        
        # Lazy-load the race prediction model on first use
        if not hasattr(self, 'race_model') or self.race_model is None:
            try:
                # Import required components from FairFace
                import torchvision.models as models
                import torch.nn as nn
                
                # Initialize the race model
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.race_model = models.resnet34(pretrained=True)
                self.race_model.fc = nn.Linear(self.race_model.fc.in_features, 18)
                models_path = os.path.join(fairface_dir, 'models')
                model_path = os.path.join(models_path, 'res34_fair_align_multi_4_20190809.pt')
                self.race_model.load_state_dict(torch.load(model_path))
                self.race_model = self.race_model.to(device)
                self.race_model.eval()
                print("FairFace race prediction model loaded successfully.")
            except Exception as e:
                print(f"Error loading race prediction model: {e}")
                return torch.tensor(0.0, device=source.device)
            
        # Define the image transformation
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process images in batches
        batch_size = source.size(0)
        loss = 0.0
        
        # Set model to evaluation mode
        self.race_model.eval()
        
        with torch.no_grad():
            for i in range(batch_size):
                # Extract and prepare single images
                src_img = trans(source[i:i+1])  # Keep batch dimension
                tgt_img = trans(target[i:i+1])  # Keep batch dimension
                
                # Get race predictions
                src_outputs = self.race_model(src_img)
                tgt_outputs = self.race_model(tgt_img)
                
                # Extract race logits from outputs (first 4 classes are for race)
                src_race_logits = src_outputs[:, :4]
                tgt_race_logits = tgt_outputs[:, :4]
                
                # Get probabilities
                src_race_probs = F.softmax(src_race_logits, dim=1)
                tgt_race_probs = F.softmax(tgt_race_logits, dim=1)
                
                # Calculate KL divergence loss between distributions
                kl_loss = F.kl_div(
                    F.log_softmax(tgt_race_logits, dim=1),
                    src_race_probs,
                    reduction='batchmean'
                )
                
                # Calculate L1 loss between probability distributions
                l1_loss = F.l1_loss(tgt_race_probs, src_race_probs)
                
                # Combine losses
                img_loss = 0.5 * kl_loss + 0.5 * l1_loss
                loss += img_loss
                
        # Average loss over batch
        if batch_size > 0:
            loss = loss / batch_size
            
        return loss
