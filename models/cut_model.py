import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch.nn.functional as F
import os


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

        # Add options for lips and teeth preservation
        parser.add_argument('--lambda_lips', type=float, default=2.0, help='weight for lips shape and color preservation loss')
        parser.add_argument('--lambda_teeth', type=float, default=2.0, help='weight for teeth preservation loss')

        # Add option for background preservation
        parser.add_argument('--lambda_background', type=float, default=2.0, help='weight for background content and color preservation loss')

        # Face preservation parameters
        parser.add_argument('--face_preservation', type=util.str2bool, default=False, 
                           help='preserve face features during stage 2 training')
        parser.add_argument('--lambda_face_preservation', type=float, default=10.0,
                           help='weight for face preservation loss')
        parser.add_argument('--stage1_checkpoint', type=str, default='',
                           help='path to stage 1 model checkpoint for face preservation')
        
        parser.set_defaults(pool_size=0)  # no image pooling

        # Add new options for face mask
        parser.add_argument('--include_ears_in_face', type=util.str2bool, default=False, 
                           help='include ears in face mask for preservation')
        parser.add_argument('--include_neck_in_face', type=util.str2bool, default=False, 
                           help='include neck in face mask for preservation')
        parser.add_argument('--face_mask_dilation', type=int, default=5, 
                           help='dilation of face mask in pixels to create smooth transitions')

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

        # Add face mask visualization if face preservation is enabled
        if opt.face_preservation and opt.lambda_face_preservation > 0.0:
            self.visual_names.append('face_mask_vis')

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
        # Add lips and teeth preservation losses
        if opt.lambda_lips > 0.0:
            self.loss_names.append('lips')
        if opt.lambda_teeth > 0.0:
            self.loss_names.append('teeth')
        
        # Add background preservation loss
        if opt.lambda_background > 0.0:
            self.loss_names.append('background')

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
            if (
                opt.lambda_eye > 0.0 or 
                opt.lambda_skin > 0.0 or
                opt.lambda_background > 0.0 or
                opt.lambda_lips > 0.0 or
                opt.lambda_teeth > 0.0 or
                opt.lambda_segmentation > 0.0 or
                (opt.lambda_face_preservation > 0.0 and opt.face_preservation)
            ) and opt.use_face_parser:
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

        # Add face preservation loss
        if opt.face_preservation and opt.lambda_face_preservation > 0.0:
            self.loss_names.append('face_preservation')
            
            # Load stage 1 model if checkpoint provided
            if opt.stage1_checkpoint and os.path.exists(opt.stage1_checkpoint):
                self.face_model_stage1 = networks.define_G(opt.input_nc, opt.output_nc, 
                                                          opt.ngf, opt.netG, opt.normG, 
                                                          not opt.no_dropout, opt.init_type, 
                                                          opt.init_gain, opt.no_antialias, 
                                                          opt.no_antialias_up, self.gpu_ids, opt)
                state_dict = torch.load(opt.stage1_checkpoint)
                self.face_model_stage1.load_state_dict(state_dict)
                self.face_model_stage1.eval()
                print('Loaded stage 1 face model from', opt.stage1_checkpoint)
            else:
                print('WARNING: face_preservation enabled but no stage1_checkpoint provided')

            # Create visualization of face mask for the first image in batch
            self.face_mask_vis = None

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
            
        # Create face mask visualization if face preservation is enabled
        if self.opt.face_preservation and self.opt.lambda_face_preservation > 0.0:
            face_masks = self.get_face_masks(self.real_A)
            # Create colorized visualization of the face mask
            self.face_mask_vis = self.colorize_mask(face_masks, self.real_A)

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
            
        # Calculate lips preservation loss
        if self.opt.lambda_lips > 0.0:
            self.loss_lips = self.compute_lips_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_lips
        else:
            self.loss_lips = 0.0
            
        # Calculate teeth preservation loss
        if self.opt.lambda_teeth > 0.0:
            self.loss_teeth = self.compute_teeth_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_teeth
        else:
            self.loss_teeth = 0.0

        # Calculate background preservation loss
        if self.opt.lambda_background > 0.0:
            self.loss_background = self.compute_background_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_background
        else:
            self.loss_background = 0.0

        # Add face preservation loss during stage 2 training
        if self.opt.face_preservation and hasattr(self, 'face_model_stage1'):
            # Extract face regions by combining facial component masks
            # We can use a combined mask from existing facial component extractions
            face_masks = self.get_face_masks(self.real_A)
            
            # Calculate the ratio of face area to total image area
            face_area_ratio = face_masks.mean([1, 2, 3])  # Average over C, H, W dimensions to get ratio per batch item
            
            # Initialize face preservation loss
            self.loss_face_preservation = torch.tensor(0.0, device=self.device)
            
            # Check if batch contains images with faces larger than threshold (20%)
            face_threshold = 0.20  # 20% threshold
            has_significant_face = face_area_ratio > face_threshold
            
            if has_significant_face.any():
                # Apply face mask to get face regions (only for images with significant faces)
                real_A_face = self.real_A * face_masks
                fake_B_face = self.fake_B * face_masks
                
                # Get face translation from stage 1 model
                with torch.no_grad():
                    # Forward pass with the stage 1 model
                    stage1_fake_B = self.face_model_stage1(self.real_A)
                    # Apply face mask to focus only on face regions
                    stage1_fake_B_face = stage1_fake_B * face_masks
                
                # Force current model to match stage 1 output for faces using L1 loss
                # Only compute loss for images with significant faces
                if has_significant_face.all():
                    # All images have significant faces, compute loss on full batch
                    self.loss_face_preservation = F.l1_loss(fake_B_face, stage1_fake_B_face) * self.opt.lambda_face_preservation
                else:
                    # Some images have significant faces, compute loss only on those
                    # Get indices of images with significant faces
                    indices = torch.where(has_significant_face)[0]
                    
                    # Select only the images with significant faces
                    sel_fake_B_face = fake_B_face[indices]
                    sel_stage1_fake_B_face = stage1_fake_B_face[indices]
                    
                    # Compute loss only on selected images
                    self.loss_face_preservation = F.l1_loss(sel_fake_B_face, sel_stage1_fake_B_face) * self.opt.lambda_face_preservation
                
                # Log the face area ratio for monitoring
                self.face_area_ratio = face_area_ratio.mean().item()
            else:
                # No significant faces in batch, set face area ratio for logging
                self.face_area_ratio = face_area_ratio.mean().item()
                
            # Add to total loss
            loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_eye + self.loss_skin + \
                     self.loss_segmentation + self.loss_edge + self.loss_color + self.loss_ethnicity + \
                     self.loss_lips + self.loss_teeth + self.loss_background + self.loss_face_preservation
        else:
            loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_eye + self.loss_skin + \
                     self.loss_segmentation + self.loss_edge + self.loss_color + self.loss_ethnicity + \
                     self.loss_lips + self.loss_teeth + self.loss_background

        return loss_G

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
        """Compute ethnicity and age preservation loss to keep the same race/ethnicity and age between source and target
        
        Args:
            source (tensor): Source image tensor
            target (tensor): Target image tensor
            
        Returns:
            tensor: Ethnicity and age preservation loss
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
        
        # Lazy-load the race and age prediction models on first use
        if not hasattr(self, 'fairface_model') or self.fairface_model is None:
            try:
                # Import required components from FairFace
                import torchvision.models as models
                import torch.nn as nn
                
                # Initialize the model for race, gender, and age prediction
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.fairface_model = models.resnet34(pretrained=True)
                self.fairface_model.fc = nn.Linear(self.fairface_model.fc.in_features, 18)
                models_path = os.path.join(fairface_dir, 'models')
                model_path = os.path.join(models_path, 'res34_fair_align_multi_4_20190809.pt')
                
                # Try to load the model, if it fails, use the race-only model
                try:
                    self.fairface_model.load_state_dict(torch.load(model_path))
                    self.has_full_fairface = True
                    print("FairFace full model (race, gender, age) loaded successfully.")
                except:
                    # Fallback to race-only model
                    model_path = os.path.join(models_path, 'res34_fair_align_multi_4_20190809.pt')
                    self.fairface_model.load_state_dict(torch.load(model_path))
                    self.has_full_fairface = False
                    print("FairFace race-only model loaded successfully.")
                
                self.fairface_model = self.fairface_model.to(device)
                self.fairface_model.eval()
            except Exception as e:
                print(f"Error loading FairFace model: {e}")
                return torch.tensor(0.0, device=source.device)
            
        # Define the image transformation
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process images in batches
        batch_size = source.size(0)
        race_loss = 0.0
        age_loss = 0.0
        
        # Set model to evaluation mode
        self.fairface_model.eval()
        
        with torch.no_grad():
            for i in range(batch_size):
                # Extract and prepare single images
                src_img = trans(source[i:i+1])  # Keep batch dimension
                tgt_img = trans(target[i:i+1])  # Keep batch dimension
                
                # Get predictions
                src_outputs = self.fairface_model(src_img)
                tgt_outputs = self.fairface_model(tgt_img)
                
                # Extract race logits from outputs (first 4 classes are for race)
                src_race_logits = src_outputs[:, :4]
                tgt_race_logits = tgt_outputs[:, :4]
                
                # Get race probabilities
                src_race_probs = F.softmax(src_race_logits, dim=1)
                tgt_race_probs = F.softmax(tgt_race_logits, dim=1)
                
                # Calculate race preservation loss
                race_kl_loss = F.kl_div(
                    F.log_softmax(tgt_race_logits, dim=1),
                    src_race_probs,
                    reduction='batchmean'
                )
                race_l1_loss = F.l1_loss(tgt_race_probs, src_race_probs)
                img_race_loss = 0.5 * race_kl_loss + 0.5 * race_l1_loss
                race_loss += img_race_loss
                
                # If we have the full FairFace model with age prediction
                if self.has_full_fairface:
                    # Extract age logits (indices 9-18 are for age)
                    src_age_logits = src_outputs[:, 9:18]
                    tgt_age_logits = tgt_outputs[:, 9:18]
                    
                    # Get age probabilities
                    src_age_probs = F.softmax(src_age_logits, dim=1)
                    tgt_age_probs = F.softmax(tgt_age_logits, dim=1)
                    
                    # Calculate age preservation loss
                    age_kl_loss = F.kl_div(
                        F.log_softmax(tgt_age_logits, dim=1),
                        src_age_probs,
                        reduction='batchmean'
                    )
                    age_l1_loss = F.l1_loss(tgt_age_probs, src_age_probs)
                    
                    # Calculate expected age difference
                    # Age categories: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
                    age_midpoints = torch.tensor([1.0, 6.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0], 
                                                device=src_age_probs.device)
                    
                    # Calculate expected age for source and target
                    src_expected_age = torch.sum(src_age_probs * age_midpoints, dim=1)
                    tgt_expected_age = torch.sum(tgt_age_probs * age_midpoints, dim=1)
                    
                    # Add expected age difference loss (normalized by typical age range)
                    age_diff_loss = F.l1_loss(src_expected_age, tgt_expected_age) / 75.0
                    
                    # Combine age losses
                    img_age_loss = 0.4 * age_kl_loss + 0.4 * age_l1_loss + 0.2 * age_diff_loss
                    age_loss += img_age_loss
                
        # Average losses over batch
        if batch_size > 0:
            race_loss = race_loss / batch_size
            if self.has_full_fairface:
                age_loss = age_loss / batch_size
                
        # Combine race and age losses
        if self.has_full_fairface:
            # Weight race and age losses (0.6 for race, 0.4 for age)
            final_loss = 0.6 * race_loss + 0.4 * age_loss
        else:
            # If we don't have age prediction, just use race loss
            final_loss = race_loss
            
        return final_loss

    def extract_lip_regions(self, img):
        """Extract lip regions from an image using face parsing"""
        if self.has_face_parser:
            # Get face parsing masks
            masks = self.get_face_parsing_masks(img)
            
            # Create lip mask (upper and lower lips)
            lip_mask = ((masks == 11) | (masks == 12)).float().unsqueeze(1)  # 11: u_lip, 12: l_lip
            
            # Apply mask to get lip regions
            lip_regions = img * lip_mask
            
            return lip_regions, lip_mask
        else:
            # Fallback to simple heuristic if face parser not available
            b, c, h, w = img.size()
            # Create a mask with the same dimensions as the original image
            mask = torch.zeros((b, 1, h, w), device=img.device)
            mask[:, :, h//2:3*h//4, w//3:2*w//3] = 1
            
            # Apply mask to get mouth region
            mouth_region = img * mask
            
            return mouth_region, mask

    def extract_teeth_regions(self, img):
        """Extract teeth/inner mouth regions using face parsing"""
        if self.has_face_parser:
            # Get face parsing masks
            masks = self.get_face_parsing_masks(img)
            
            # Create mouth mask
            mouth_mask = (masks == 10).float().unsqueeze(1)  # 10: mouth
            
            # Apply mask to get inner mouth/teeth regions
            teeth_regions = img * mouth_mask
            
            return teeth_regions, mouth_mask
        else:
            # Simple fallback
            b, c, h, w = img.size()
            # Create a mask with the same dimensions as the original image
            mask = torch.zeros((b, 1, h, w), device=img.device)
            mask[:, :, h//2:3*h//4, w//3+w//12:2*w//3-w//12] = 1
            
            # Apply mask to get teeth region
            teeth_region = img * mask
            
            return teeth_region, mask

    def compute_lips_preservation_loss(self, src, tgt):
        """Calculate loss for preserving lip shape and color"""
        src_lips, src_mask = self.extract_lip_regions(src)
        tgt_lips, tgt_mask = self.extract_lip_regions(tgt)
        
        # Create combined mask - only consider pixels where both source and target detected lips
        combined_mask = src_mask * tgt_mask
        
        # If no lip pixels are detected, return zero loss
        if combined_mask.sum() < 1.0:
            return torch.tensor(0.0, device=self.device)
        
        # Apply mask
        src_lips_masked = src_lips * combined_mask
        tgt_lips_masked = tgt_lips * combined_mask
        
        # Calculate color statistics
        src_mean = torch.sum(src_lips_masked, dim=[2, 3]) / (combined_mask.sum() + 1e-6)
        
        # Calculate shape and color preservation losses
        color_loss = F.l1_loss(src_mean, tgt_mean)
        
        # For shape preservation, we use both texture and edge losses
        texture_loss = F.l1_loss(src_lips_masked, tgt_lips_masked)
        
        # Detect edges in the lip regions for shape analysis
        src_lips_edges, _, _ = self.detect_edges(src_lips_masked)
        tgt_lips_edges, _, _ = self.detect_edges(tgt_lips_masked)
        edge_loss = F.l1_loss(src_lips_edges, tgt_lips_edges)
        
        # Combine losses with appropriate weights
        return color_loss * 0.3 + texture_loss * 0.4 + edge_loss * 0.3

    def compute_teeth_preservation_loss(self, src, tgt):
        """Calculate loss for preserving teeth appearance"""
        src_teeth, src_mask = self.extract_teeth_regions(src)
        tgt_teeth, tgt_mask = self.extract_teeth_regions(tgt)
        
        # Create combined mask
        combined_mask = src_mask * tgt_mask
        
        # If no teeth/mouth pixels are detected, return zero loss
        if combined_mask.sum() < 1.0:
            return torch.tensor(0.0, device=self.device)
        
        # Apply mask
        src_teeth_masked = src_teeth * combined_mask
        tgt_teeth_masked = tgt_teeth * combined_mask
        
        # For teeth, brightness and whiteness are important
        # Calculate brightness
        src_brightness = torch.mean(src_teeth_masked, dim=1, keepdim=True)
        tgt_brightness = torch.mean(tgt_teeth_masked, dim=1, keepdim=True)
        
        # Calculate whiteness (how close to white)
        # White has equal RGB values close to 1, so we penalize deviation from this pattern
        src_whiteness = torch.std(src_teeth_masked, dim=1, keepdim=True)  # Lower std means more equal RGB values
        tgt_whiteness = torch.std(tgt_teeth_masked, dim=1, keepdim=True)
        
        # Calculate teeth preservation losses
        brightness_loss = F.l1_loss(src_brightness, tgt_brightness)
        whiteness_loss = F.l1_loss(src_whiteness, tgt_whiteness)
        texture_loss = F.l1_loss(src_teeth_masked, tgt_teeth_masked)
        
        # Combine losses
        return brightness_loss * 0.3 + whiteness_loss * 0.3 + texture_loss * 0.4

    def compute_background_preservation_loss(self, source, target):
        """Calculate loss for preserving background content and color"""
        if not self.has_face_parser:
            return torch.tensor(0.0, device=self.device)
            
        # Get background masks using the face parser (background label is 0)
        source_background = self.extract_background_regions(source)
        target_background = self.extract_background_regions(target)
        
        # If no background is found, return zero loss
        if source_background is None or target_background is None:
            return torch.tensor(0.0, device=self.device)
            
        # Calculate L1 loss for color preservation in background regions
        color_loss = F.l1_loss(source_background, target_background)
        
        # Calculate perceptual loss for content preservation
        # We use a combination of low-level features (color) and high-level features (content/structure)
        background_mask = self.get_background_mask(source)
        
        # Apply mask to both source and target images
        source_masked = source * background_mask
        target_masked = target * background_mask
        
        # Calculate structure similarity for the background regions
        ssim_loss = 1 - self.calculate_ssim(source_masked, target_masked)
        
        # Combine losses with appropriate weights
        total_loss = color_loss * 0.6 + ssim_loss * 0.4
        
        return total_loss
        
    def extract_background_regions(self, img):
        """Extract background regions from an image using face parsing"""
        # Get face parsing masks
        masks = self.get_face_parsing_masks(img)
        
        # Create background mask (where label is 0)
        background_mask = (masks == 0).float()
        
        # Check if there's enough background
        if background_mask.sum() < 100:  # Arbitrary threshold, adjust as needed
            return None
            
        # Expand mask to match image channels
        background_mask = background_mask.unsqueeze(1).expand(-1, img.size(1), -1, -1)
        
        # Apply mask to extract background regions
        background_regions = img * background_mask
        
        return background_regions
        
    def get_background_mask(self, img):
        """Get background mask from face parsing"""
        # Get face parsing masks
        masks = self.get_face_parsing_masks(img)
        
        # Create background mask (where label is 0)
        background_mask = (masks == 0).float()
        
        # Expand mask to match image channels
        background_mask = background_mask.unsqueeze(1).expand(-1, img.size(1), -1, -1)
        
        return background_mask
        
    def calculate_ssim(self, img1, img2):
        """Calculate structural similarity between two images"""
        # Convert to luminance
        cs = torch.ones_like(img1[:, 0:1, :, :]) * 0.45
        lum1 = img1[:, 0:1, :, :] * 0.2126 + img1[:, 1:2, :, :] * 0.7152 + img1[:, 2:3, :, :] * 0.0722
        lum2 = img2[:, 0:1, :, :] * 0.2126 + img2[:, 1:2, :, :] * 0.7152 + img2[:, 2:3, :, :] * 0.0722
        
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Calculate means and variances
        mu1 = F.avg_pool2d(lum1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(lum2, kernel_size=11, stride=1, padding=5)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(lum1 * lum1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(lum2 * lum2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(lum1 * lum2, kernel_size=11, stride=1, padding=5) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return mean SSIM
        return ssim_map.mean()

    def get_face_masks(self, img):
        """Create binary masks separating face from non-face regions using face parsing model"""
        batch_size = img.size(0)
        h, w = img.size(2), img.size(3)
        
        # Initialize empty mask
        face_mask = torch.zeros((batch_size, 1, h, w), device=img.device)
        
        # If we have face parsing capability, use it directly
        if hasattr(self, 'has_face_parser') and self.has_face_parser:
            parsing_masks = self.get_face_parsing_masks(img)
            
            # Based on the face parser configuration, define facial features
            # Map of semantic labels from config.json
            face_feature_labels = [
                1,   # skin
                2,   # nose
                3,   # eye_g (glasses)
                4,   # l_eye
                5,   # r_eye
                6,   # l_brow
                7,   # r_brow
                10,  # mouth
                11,  # u_lip
                12,  # l_lip
            ]
            
            # Optional: include ears, neck in face mask
            if hasattr(self.opt, 'include_ears_in_face') and self.opt.include_ears_in_face:
                face_feature_labels.extend([8, 9])  # left ear, right ear
            
            if hasattr(self.opt, 'include_neck_in_face') and self.opt.include_neck_in_face:
                face_feature_labels.extend([16, 17])  # neck_l, neck
            
            # Create facial features mask
            facial_features_mask = torch.zeros_like(parsing_masks, dtype=torch.bool)
            for label in face_feature_labels:
                facial_features_mask = torch.logical_or(facial_features_mask, parsing_masks == label)
            
            # Convert to float and add channel dimension
            face_mask = facial_features_mask.float().unsqueeze(1)
        else:
            # Fallback to component-based approach if face parser not available
            # Add eye regions to mask
            if hasattr(self, 'extract_eye_regions'):
                _, eye_mask = self.extract_eye_regions(img)
                face_mask = torch.max(face_mask, eye_mask)
            
            # Add skin regions to mask
            if hasattr(self, 'extract_skin_regions'):
                _, skin_mask = self.extract_skin_regions(img)
                face_mask = torch.max(face_mask, skin_mask)
            
            # Add lip regions to mask
            if hasattr(self, 'extract_lip_regions'):
                _, lip_mask = self.extract_lip_regions(img)
                face_mask = torch.max(face_mask, lip_mask)
            
            # Add teeth regions to mask
            if hasattr(self, 'extract_teeth_regions'):
                _, teeth_mask = self.extract_teeth_regions(img)
                face_mask = torch.max(face_mask, teeth_mask)
        
        # Optionally dilate the mask to include a small border around the face
        if hasattr(self.opt, 'face_mask_dilation') and self.opt.face_mask_dilation > 0:
            kernel_size = self.opt.face_mask_dilation * 2 + 1
            padding = self.opt.face_mask_dilation
            face_mask = F.max_pool2d(face_mask, kernel_size=kernel_size, 
                                   stride=1, padding=padding)
        
        return face_mask

    def colorize_mask(self, mask, original_img):
        """Create a color visualization of the face mask overlaid on the original image
        
        Args:
            mask: Binary mask tensor [B, 1, H, W]
            original_img: Original image tensor [B, C, H, W]
            
        Returns:
            Visualization tensor [B, 3, H, W]
        """
        # Use a distinguishable color for the face mask overlay (light green)
        batch_size, _, h, w = mask.size()
        
        # Create an RGB highlight for the mask
        highlight_color = torch.tensor([0.0, 1.0, 0.3], device=mask.device).view(1, 3, 1, 1)
        mask_rgb = highlight_color.expand(batch_size, 3, h, w) * mask.expand(-1, 3, -1, -1)
        
        # Create a semi-transparent overlay
        alpha = 0.5
        overlay = original_img * (1.0 - alpha * mask.expand(-1, 3, -1, -1)) + mask_rgb * alpha
        
        # Add a border around the mask
        kernel_size = 3
        padding = kernel_size // 2
        dilated_mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        edge_mask = (dilated_mask - mask).expand(-1, 3, -1, -1)
        
        # Add bright border around the mask (yellow)
        border_color = torch.tensor([1.0, 1.0, 0.0], device=mask.device).view(1, 3, 1, 1)
        border_rgb = border_color.expand(batch_size, 3, h, w) * edge_mask
        
        # Add border to the overlay
        overlay = overlay + border_rgb
        
        # Calculate face area percentage for each image in batch
        face_area_ratio = mask.mean([1, 2, 3]) * 100  # Convert to percentage
        
        # Add text overlay with face area percentage
        # We'll create this by drawing directly on the first few rows of the image
        for i in range(batch_size):
            # Format text: "Face: XX.X%" with threshold indicator
            ratio_value = face_area_ratio[i].item()
            threshold_indicator = "✓" if ratio_value >= 20.0 else "✗"
            
            # Set text color based on threshold (white or yellow)
            text_color = torch.tensor([1.0, 1.0, 1.0], device=mask.device) if ratio_value >= 20.0 else torch.tensor([1.0, 1.0, 0.0], device=mask.device)
            
            # Create text background (dark rectangle at top)
            overlay[i, :, :20, :200] = torch.tensor([0.0, 0.0, 0.0], device=mask.device).view(3, 1, 1)
            
            # Draw simple text representation using rectangles (limited by tensor operations)
            # Display "Face: XX.X% [✓/✗]"
            # We'll make a simple colored indicator in the top-left corner
            indicator_color = torch.tensor([0.0, 1.0, 0.0], device=mask.device) if ratio_value >= 20.0 else torch.tensor([1.0, 0.0, 0.0], device=mask.device)
            
            # Draw colored indicator box
            overlay[i, :, 5:15, 5:15] = indicator_color.view(3, 1, 1)
            
            # Draw percentage text (simple colored box to represent text)
            # This is a basic approach since we can't easily render text in tensors
            overlay[i, :, 5:15, 20:100] = text_color.view(3, 1, 1) * 0.8
        
        return overlay
