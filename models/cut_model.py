import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch.nn.functional as F
import os
import cv2
import lpips


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
        
        # Add LPIPS loss option
        parser.add_argument('--lambda_lpips', type=float, default=0.0, help='weight for LPIPS perceptual loss')
        parser.add_argument('--lpips_net', type=str, default='vgg', choices=['alex', 'vgg'], help='network to use for LPIPS loss')
        
        # Add attention mechanism options
        parser.add_argument('--attention_layers', type=str, default="16,20", 
                          help='apply attention on which generator layers (comma-separated layer indices)')
        parser.add_argument('--attention_heads', type=int, default=4,
                          help='number of attention heads in multihead attention')
        parser.add_argument('--use_attention', type=util.str2bool, default=False,
                          help='enable attention mechanism in generator')
        
        # Add new options for eye color and skin tone preservation
        parser.add_argument('--lambda_eye', type=float, default=0.0, help='weight for eye color preservation loss')
        parser.add_argument('--use_face_parser', type=util.str2bool, default=True, help='use face parsing model for precise feature extraction')
        parser.add_argument('--lambda_segmentation', type=float, default=0.0, help='weight for segmentation consistency loss')

        # Add new options for edge preservation and color consistency
        parser.add_argument('--lambda_edge', type=float, default=0.0, help='weight for edge preservation loss')
        parser.add_argument('--lambda_color_consistency', type=float, default=0.0, help='weight for color consistency loss to prevent bleeding')
        parser.add_argument('--edge_threshold', type=float, default=0.05, help='threshold for detecting important edges')

        # Add option for ethnicity preservation
        parser.add_argument('--lambda_ethnicity', type=float, default=0.0, help='weight for ethnicity preservation loss')


        # Add option for background preservation
        parser.add_argument('--lambda_background', type=float, default=0.0, help='weight for background content and color preservation loss')

        # Face preservation parameters
        parser.add_argument('--face_preservation', type=util.str2bool, default=False, 
                           help='preserve face features during stage 2 training')
        parser.add_argument('--lambda_face_preservation', type=float, default=10.0,
                           help='weight for face preservation loss')
        parser.add_argument('--stage1_checkpoint', type=str, default='',
                           help='path to stage 1 model checkpoint for face preservation')
        
        # Add non-face preservation parameters
        parser.add_argument('--non_face_preservation', type=util.str2bool, default=False,
                           help='preserve non-face regions during training')
        parser.add_argument('--lambda_non_face_preservation', type=float, default=10.0,
                           help='weight for non-face preservation loss')
        
        parser.set_defaults(pool_size=0)  # no image pooling

        # Add new options for face mask
        parser.add_argument('--include_ears_in_face', type=util.str2bool, default=False, 
                           help='include ears in face mask for preservation')
        parser.add_argument('--include_neck_in_face', type=util.str2bool, default=False, 
                           help='include neck in face mask for preservation')
        parser.add_argument('--face_mask_dilation', type=int, default=5, 
                           help='dilation of face mask in pixels to create smooth transitions')

        # Add new options for non-face color preservation
        parser.add_argument('--lambda_non_face_color', type=float, default=0.0, 
                           help='weight for preserving colors in non-face regions')

        # Add InsightFace-based identity preservation options
        parser.add_argument('--lambda_identity', type=float, default=0.0,
                           help='weight for identity preservation loss using InsightFace')
        parser.add_argument('--use_insightface', type=util.str2bool, default=True,
                           help='use InsightFace for identity preservation')
        parser.add_argument('--insightface_model', type=str, default='buffalo_l',
                           choices=['buffalo_l', 'buffalo_m', 'buffalo_s'],
                           help='InsightFace model variant to use')

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

        # Add LPIPS loss name if enabled
        if opt.lambda_lpips > 0.0:
            self.loss_names.append('lpips')

        # Add identity preservation loss name if enabled
        if opt.lambda_identity > 0.0 and opt.use_insightface:
            self.loss_names.append('identity')

        # Add face mask visualization if face preservation is enabled
        if opt.face_preservation and opt.lambda_face_preservation > 0.0:
            self.visual_names.append('face_mask_vis')

        # Add new losses for eye color and skin tone preservation
        if opt.lambda_eye > 0.0:
            self.loss_names.append('eye')

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
            
            # Initialize LPIPS loss if enabled
            if opt.lambda_lpips > 0.0:
                self.criterionLPIPS = lpips.LPIPS(net=opt.lpips_net).to(self.device)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            # Initialize face parser if needed
            if (
                opt.lambda_eye > 0.0 or 
                opt.lambda_background > 0.0 or
                opt.lambda_segmentation > 0.0 or
                opt.lambda_non_face_color > 0.0 or
                (opt.lambda_face_preservation > 0.0 and opt.face_preservation) or
                (opt.lambda_non_face_preservation > 0.0 and opt.non_face_preservation)
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
            
        # Add non-face preservation loss
        if opt.non_face_preservation and opt.lambda_non_face_preservation > 0.0:
            self.loss_names.append('non_face_preservation')
            
            # Create visualization of non-face mask
            self.non_face_mask_vis = None

        # Initialize InsightFace for identity preservation
        self.insightface_model = None
        if opt.lambda_identity > 0.0 and opt.use_insightface:
            try:
                from insightface.app import FaceAnalysis
                from insightface.app.common import Face
                import cv2
                import numpy as np
                
                # Initialize InsightFace
                self.insightface_model = FaceAnalysis(name=opt.insightface_model)
                self.insightface_model.prepare(ctx_id=0, det_size=(640, 640))
                print("InsightFace model loaded successfully")
                
                # Store face analysis results
                self.face_analysis_cache = {}
            except ImportError:
                print("Warning: InsightFace not found. Identity preservation will be disabled.")
                self.insightface_model = None
                opt.use_insightface = False

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
            
        # Calculate LPIPS perceptual loss
        if self.opt.lambda_lpips > 0.0:
            self.loss_lpips = self.criterionLPIPS(self.real_A, self.fake_B).mean() * self.opt.lambda_lpips
        else:
            self.loss_lpips = 0.0

        # Calculate background preservation loss
        if self.opt.lambda_background > 0.0:
            self.loss_background = self.compute_background_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_background
        else:
            self.loss_background = 0.0

        # Calculate identity preservation loss using InsightFace
        if self.opt.lambda_identity > 0.0 and self.opt.use_insightface:
            self.loss_identity = self.compute_identity_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_identity
        else:
            self.loss_identity = 0.0

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
            face_threshold = 0.10  # 10% threshold
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
            loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_eye +  \
                     self.loss_segmentation + self.loss_edge + self.loss_color + self.loss_ethnicity + \
                     self.loss_background + self.loss_face_preservation + self.loss_lpips
        else:
            loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_eye + \
                     self.loss_segmentation + self.loss_edge + self.loss_color + self.loss_ethnicity + \
                     self.loss_background + self.loss_lpips

        # Add non-face preservation loss
        if self.opt.non_face_preservation and self.opt.lambda_non_face_preservation > 0.0:
            self.loss_non_face_preservation = self.compute_non_face_preservation_loss(self.real_A, self.fake_B)
            loss_G += self.loss_non_face_preservation

        # Add non-face color preservation loss
        if self.opt.lambda_non_face_color > 0.0:
            self.loss_names.append('non_face_color')

        # Calculate non-face color preservation loss
        if self.opt.lambda_non_face_color > 0.0:
            self.loss_non_face_color = self.compute_non_face_color_preservation_loss(self.real_A, self.fake_B) * self.opt.lambda_non_face_color
        else:
            self.loss_non_face_color = 0.0
            
        # Add to total loss
        loss_G += self.loss_non_face_color

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

    def compute_non_face_color_preservation_loss(self, source, target):
        """Calculate loss for preserving colors in non-face regions
        
        Args:
            source (tensor): Source image tensor [B, C, H, W]
            target (tensor): Target image tensor [B, C, H, W]
            
        Returns:
            tensor: Non-face color preservation loss
        """
        # Get face masks
        face_masks = self.get_face_masks(source)
        
        # Invert masks to get non-face regions (add small epsilon to avoid division by zero)
        non_face_masks = 1.0 - face_masks
        
        # Skip the loss if there are no significant non-face regions
        if non_face_masks.mean() < 0.1:  # Less than 10% of the image is non-face
            return torch.tensor(0.0, device=source.device)
        
        # Apply masks to focus only on non-face regions
        source_non_face = source * non_face_masks
        target_non_face = target * non_face_masks
        
        # Calculate color preservation loss (L1 loss)
        color_loss = F.l1_loss(source_non_face, target_non_face, reduction='sum') / (non_face_masks.sum() + 1e-6)
        
        # Additionally calculate color statistics preservation
        # Extract per-channel mean and standard deviation for non-face regions
        source_mean = []
        target_mean = []
        source_std = []
        target_std = []
        
        for c in range(source.size(1)):  # For each channel
            # Calculate weighted mean and std using the mask as weights
            src_channel = source[:, c:c+1, :, :]
            tgt_channel = target[:, c:c+1, :, :]
            
            # Weighted means
            src_mean_c = (src_channel * non_face_masks).sum(dim=[2, 3]) / (non_face_masks.sum(dim=[2, 3]) + 1e-6)
            tgt_mean_c = (tgt_channel * non_face_masks).sum(dim=[2, 3]) / (non_face_masks.sum(dim=[2, 3]) + 1e-6)
            
            # Weighted standard deviations
            src_diff_sq = ((src_channel - src_mean_c.view(-1, 1, 1, 1)) ** 2) * non_face_masks
            tgt_diff_sq = ((tgt_channel - tgt_mean_c.view(-1, 1, 1, 1)) ** 2) * non_face_masks
            
            src_std_c = torch.sqrt(src_diff_sq.sum(dim=[2, 3]) / (non_face_masks.sum(dim=[2, 3]) + 1e-6))
            tgt_std_c = torch.sqrt(tgt_diff_sq.sum(dim=[2, 3]) / (non_face_masks.sum(dim=[2, 3]) + 1e-6))
            
            source_mean.append(src_mean_c)
            target_mean.append(tgt_mean_c)
            source_std.append(src_std_c)
            target_std.append(tgt_std_c)
        
        # Combine channel statistics
        source_mean = torch.cat(source_mean, dim=1)
        target_mean = torch.cat(target_mean, dim=1)
        source_std = torch.cat(source_std, dim=1)
        target_std = torch.cat(target_std, dim=1)
        
        # Calculate mean and standard deviation preservation losses
        mean_loss = F.l1_loss(source_mean, target_mean)
        std_loss = F.l1_loss(source_std, target_std)
        
        # Combine losses: pixel-wise color loss + color statistics loss
        total_loss = 0.7 * color_loss + 0.2 * mean_loss + 0.1 * std_loss
        
        return total_loss

    def compute_non_face_preservation_loss(self, source, target):
        """Calculate loss for preserving everything except face regions
        
        Args:
            source (tensor): Source image tensor [B, C, H, W]
            target (tensor): Target image tensor [B, C, H, W]
            
        Returns:
            tensor: Non-face preservation loss
        """
        # Get face masks
        face_masks = self.get_face_masks(source)
        
        # Invert masks to get non-face regions
        non_face_masks = 1.0 - face_masks
        
        # Skip the loss if there are no significant non-face regions
        if non_face_masks.mean() < 0.1:  # Less than 10% of the image is non-face
            return torch.tensor(0.0, device=source.device)
        
        # Apply masks to focus only on non-face regions
        source_non_face = source * non_face_masks
        target_non_face = target * non_face_masks
        
        # Calculate L1 loss for non-face regions
        # Scale by the size of the non-face region to normalize the loss
        loss = F.l1_loss(source_non_face, target_non_face, reduction='sum') / (non_face_masks.sum() + 1e-6)
        
        # Apply weight from options
        weighted_loss = loss * self.opt.lambda_non_face_preservation
        
        # Create visualization of non-face mask
        if self.non_face_mask_vis is None:
            self.non_face_mask_vis = self.colorize_mask(non_face_masks, source)
        
        return weighted_loss

    def compute_identity_preservation_loss(self, source, target):
        """Calculate identity preservation loss using InsightFace
        
        Args:
            source (tensor): Source image tensor [B, C, H, W]
            target (tensor): Target image tensor [B, C, H, W]
            
        Returns:
            tensor: Identity preservation loss
        """
        if self.insightface_model is None:
            return torch.tensor(0.0, device=source.device)
            
        batch_size = source.size(0)
        total_loss = 0.0
        
        # Process each image in the batch
        for i in range(batch_size):
            # Convert tensors to numpy arrays (detach to avoid gradient computation)
            src_img = source[i].detach().permute(1, 2, 0).cpu().numpy()
            tgt_img = target[i].detach().permute(1, 2, 0).cpu().numpy()
            
            # Convert from [-1,1] to [0,255] range
            src_img = ((src_img + 1) * 127.5).astype(np.uint8)
            tgt_img = ((tgt_img + 1) * 127.5).astype(np.uint8)
            
            # Convert to BGR for OpenCV
            src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
            tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_RGB2BGR)
            
            # First attempt at face detection
            src_faces = self.insightface_model.get(src_img)
            tgt_faces = self.insightface_model.get(tgt_img)
            
            # If face detection fails, try with resized image on white background
            if len(src_faces) == 0 or len(tgt_faces) == 0:
                # Create white backgrounds
                h, w = src_img.shape[:2]
                src_white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
                tgt_white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
                
                # Resize images to 75% of original size
                new_h, new_w = int(h * 0.6), int(w * 0.6)
                src_resized = cv2.resize(src_img, (new_w, new_h))
                tgt_resized = cv2.resize(tgt_img, (new_w, new_h))
                
                # Calculate padding to center the resized images
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                
                # Paste resized images onto white backgrounds
                src_white_bg[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = src_resized
                tgt_white_bg[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = tgt_resized
                
                # Try face detection again
                if len(src_faces) == 0:
                    src_faces = self.insightface_model.get(src_white_bg)
                if len(tgt_faces) == 0:
                    tgt_faces = self.insightface_model.get(tgt_white_bg)
            
            # If we still have faces to compare
            if len(src_faces) > 0 and len(tgt_faces) > 0:
                # Get the largest face in each image
                src_face = max(src_faces, key=lambda x: x.bbox[2] * x.bbox[3])
                tgt_face = max(tgt_faces, key=lambda x: x.bbox[2] * x.bbox[3])
                
                # Calculate cosine similarity between face embeddings
                similarity = np.dot(src_face.embedding, tgt_face.embedding) / (
                    np.linalg.norm(src_face.embedding) * np.linalg.norm(tgt_face.embedding)
                )
                
                # Convert similarity to loss (1 - similarity)
                # Higher similarity means lower loss
                face_loss = 1.0 - similarity
                
                # Add to total loss
                total_loss += face_loss
            else:
                # If we still can't detect faces, add maximum loss
                total_loss += 1.0
        
        # Average loss over batch
        if batch_size > 0:
            total_loss = total_loss / batch_size
            
        return torch.tensor(total_loss, device=source.device)
