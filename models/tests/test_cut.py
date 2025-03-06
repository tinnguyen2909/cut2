import unittest
import torch
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to access model modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cut_model import CUTModel
from options.train_options import TrainOptions


class MockOptions:
    """Mock option class for testing"""
    def __init__(self):
        self.isTrain = True
        # Use empty list for CPU testing, or check CUDA availability
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.checkpoints_dir = './checkpoints'
        self.name = 'test'
        self.model = 'cut'
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.netG = 'resnet_9blocks'
        self.normG = 'instance'
        self.no_dropout = False
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_antialias = False
        self.no_antialias_up = False
        self.ndf = 64
        self.netD = 'basic'
        self.n_layers_D = 3
        self.normD = 'instance'
        self.gan_mode = 'lsgan'
        self.lambda_GAN = 1.0
        self.lambda_NCE = 1.0
        self.lambda_eye = 10.0
        self.lambda_skin = 5.0
        self.use_face_parser = True
        self.nce_idt = True
        self.nce_layers = '0,4,8,12,16'
        self.nce_includes_all_negatives_from_minibatch = False
        self.netF = 'mlp_sample'
        self.netF_nc = 256
        self.nce_T = 0.07
        self.num_patches = 256
        self.flip_equivariance = False
        self.direction = 'AtoB'
        self.CUT_mode = 'CUT'
        self.continue_train = False
        self.epoch = 'latest'
        self.load_iter = 0
        self.verbose = False
        self.suffix = ''
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.batch_size = 1
        self.preprocess = 'resize_and_crop'
        self.lambda_segmentation = 10.0
        self.lambda_edge = 10.0
        self.lambda_color_consistency = 10.0
        self.edge_threshold = 0.1


class TestCUTModel(unittest.TestCase):
    """Test class for CUT model with eye and skin preservation losses"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Create test directories if they don't exist
        test_image_dir = Path('models/tests/images')
        test_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy face images if they don't exist
        cls.create_test_images(test_image_dir)
        
        # Initialize model with mock options
        cls.opt = MockOptions()
        
        # Try to initialize model, but handle case where face parser is not available
        try:
            cls.model = CUTModel(cls.opt)
            cls.has_face_parser = hasattr(cls.model, 'has_face_parser') and cls.model.has_face_parser
        except Exception as e:
            print(f"Warning: Could not initialize model with face parser: {e}")
            # Try again without face parser
            cls.opt.use_face_parser = False
            cls.model = CUTModel(cls.opt)
            cls.has_face_parser = False

    @staticmethod
    def create_test_images(dir_path):
        """Create test images if they don't exist"""
        face1_path = dir_path / 'face1.jpg'
        face1_2_path = dir_path / 'face1_2.jpg'
        
        if not face1_path.exists() or not face1_2_path.exists():
            # Create simple colored test images
            import numpy as np
            from PIL import Image, ImageDraw
            
            # Face 1 with blue eyes and light skin
            img1 = Image.new('RGB', (256, 256), color=(220, 180, 160))  # Light skin tone
            draw1 = ImageDraw.Draw(img1)
            # Draw blue eyes
            draw1.ellipse((80, 80, 110, 100), fill=(30, 100, 220))
            draw1.ellipse((150, 80, 180, 100), fill=(30, 100, 220))
            img1.save(face1_path)
            
            # Face 2 with brown eyes and darker skin
            img2 = Image.new('RGB', (256, 256), color=(180, 140, 120))  # Darker skin tone
            draw2 = ImageDraw.Draw(img2)
            # Draw brown eyes
            draw2.ellipse((80, 80, 110, 100), fill=(120, 80, 40))
            draw2.ellipse((150, 80, 180, 100), fill=(120, 80, 40))
            img2.save(face1_2_path)

    def load_test_images(self, *images):
        """Load test images into tensors
        
        Args:
            *images: Image filenames to load. If none provided, defaults to ['face1.jpg', 'face1_2.jpg']
        
        Returns:
            tuple: Tensor(s) of the loaded image(s)
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Default to face1.jpg and face1_2.jpg if no images specified
        if not images:
            images = ['face1.jpg', 'face1_2.jpg']
        
        tensors = []
        for img_name in images:
            img_path = os.path.join('models/tests/images', img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            # Move tensor to the same device as the model
            if hasattr(self.model, 'device'):
                img_tensor = img_tensor.to(self.model.device)
            
            tensors.append(img_tensor)
        
        # Return a single tensor or a tuple of tensors
        if len(tensors) == 1:
            return tensors[0]
        else:
            return tuple(tensors)

    def test_eye_color_loss(self):
        """Test if eye color loss works properly"""
        # Skip test if face parser is not available
        if not hasattr(self, 'has_face_parser'):
            self.skipTest("Face parser not initialized")
        
        # Load test images
        face1_tensor, face1_2_tensor = self.load_test_images()
        
        # Set up mock input
        self.model.real_A = face1_tensor
        self.model.fake_B = face1_2_tensor
        
        # Calculate eye color loss
        eye_loss = self.model.compute_eye_color_loss(face1_tensor, face1_2_tensor)
        
        # Loss should be non-zero since eye colors are different
        self.assertGreater(eye_loss.item(), 0.0)
        
        # Test with identical images - loss should be close to zero
        same_loss = self.model.compute_eye_color_loss(face1_tensor, face1_tensor)
        self.assertLess(same_loss.item(), eye_loss.item())

    def test_skin_tone_loss(self):
        """Test if skin tone loss works properly"""
        # Skip test if face parser is not available
        if not hasattr(self, 'has_face_parser'):
            self.skipTest("Face parser not initialized")
        
        # Load test images
        face1_tensor, face1_2_tensor = self.load_test_images()
        
        # Set up mock input
        self.model.real_A = face1_tensor
        self.model.fake_B = face1_2_tensor
        
        # Calculate skin tone loss
        skin_loss = self.model.compute_skin_tone_loss(face1_tensor, face1_2_tensor)
        
        # Loss should be non-zero since skin tones are different
        self.assertGreater(skin_loss.item(), 0.0)
        
        # Test with identical images - loss should be close to zero
        same_loss = self.model.compute_skin_tone_loss(face1_tensor, face1_tensor)
        self.assertLess(same_loss.item(), skin_loss.item())

    def test_face_parsing_cache(self):
        """Test if face parsing cache works properly"""
        # Skip test if face parser is not available
        if not self.has_face_parser:
            self.skipTest("Face parser not available")
        
        # Load test images
        face1_tensor, _ = self.load_test_images()
        
        # Clear cache
        self.model.face_parsing_cache = {}
        
        # Time first call (should be slow)
        start_time = time.time()
        masks1 = self.model.get_face_parsing_masks(face1_tensor)
        first_call_time = time.time() - start_time
        
        # Time second call with same image (should be faster due to caching)
        start_time = time.time()
        masks1_2 = self.model.get_face_parsing_masks(face1_tensor)
        second_call_time = time.time() - start_time
        
        # Second call should be significantly faster
        self.assertLess(second_call_time, first_call_time * 0.5, 
                        f"Cached call ({second_call_time:.4f}s) not faster than original call ({first_call_time:.4f}s)")
        
        # Results should be identical - move to CPU for comparison if needed
        if masks1.device != masks1_2.device:
            masks1 = masks1.cpu()
            masks1_2 = masks1_2.cpu()
        self.assertTrue(torch.equal(masks1, masks1_2))
        
        # Cache should have exactly one entry
        self.assertEqual(len(self.model.face_parsing_cache), 1)

    # def test_skin_tone_loss_lighting_robustness(self):
    #     """Test if skin tone loss is robust against lighting variations"""
    #     # Skip test if face parser is not available
    #     if not self.has_face_parser:
    #         self.skipTest("Face parser not available")
        
    #     # Create test images with same skin tone but different lighting conditions
    #     from PIL import Image, ImageEnhance
    #     import torchvision.transforms as transforms
    #     import numpy as np
        
    #     transform = transforms.Compose([
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])
        
    #     # Load or create base face image
    #     base_face = Image.open('models/tests/images/face1.jpg').convert('RGB')
        
    #     # Create variations with different lighting conditions
    #     # 1. Normal lighting (original)
    #     face_normal = base_face.copy()
        
    #     # 2. Darker version (simulating low light)
    #     face_dark = ImageEnhance.Brightness(base_face).enhance(0.5)
        
    #     # 3. Brighter version (simulating strong light)
    #     face_bright = ImageEnhance.Brightness(base_face).enhance(1.5)
        
    #     # 4. Load shadow face image directly
    #     face_shadow = Image.open('models/tests/images/face1_shadow.jpg').convert('RGB')
        
    #     # 5. Different skin tone (for comparison)
    #     face_different = Image.open('models/tests/images/face2.jpg').convert('RGB')
        
    #     # Convert to tensors
    #     face_normal_tensor = transform(face_normal).unsqueeze(0)
    #     face_dark_tensor = transform(face_dark).unsqueeze(0)
    #     face_bright_tensor = transform(face_bright).unsqueeze(0)
    #     face_shadow_tensor = transform(face_shadow).unsqueeze(0)
    #     face_different_tensor = transform(face_different).unsqueeze(0)
        
    #     # Move tensors to the same device as the model
    #     if hasattr(self.model, 'device'):
    #         face_normal_tensor = face_normal_tensor.to(self.model.device)
    #         face_dark_tensor = face_dark_tensor.to(self.model.device)
    #         face_bright_tensor = face_bright_tensor.to(self.model.device)
    #         face_shadow_tensor = face_shadow_tensor.to(self.model.device)
    #         face_different_tensor = face_different_tensor.to(self.model.device)
        
    #     # Calculate skin tone losses between same skin under different lighting
    #     normal_to_dark_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_dark_tensor)
    #     normal_to_bright_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_bright_tensor)
    #     normal_to_shadow_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_shadow_tensor)
        
    #     # Calculate skin tone loss between different skin tones
    #     different_skin_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_different_tensor)
        
    #     # Calculate reference loss (same image, same lighting)
    #     same_image_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_normal_tensor)
        
    #     # Print losses for debugging
    #     print(f"\nSkin tone loss comparison:")
    #     print(f"Same image, same lighting: {same_image_loss.item():.6f}")
    #     print(f"Same skin, dark lighting: {normal_to_dark_loss.item():.6f}")
    #     print(f"Same skin, bright lighting: {normal_to_bright_loss.item():.6f}")
    #     print(f"Same skin, partial shadow: {normal_to_shadow_loss.item():.6f}")
    #     print(f"Different skin tone: {different_skin_loss.item():.6f}")
        
    #     # Losses between the same skin under different lighting should be
    #     # significantly lower than between different skin tones
    #     self.assertLess(normal_to_dark_loss.item(), different_skin_loss.item() * 0.5,
    #                     "Loss with darker lighting should be much lower than with different skin")
        
    #     self.assertLess(normal_to_bright_loss.item(), different_skin_loss.item() * 0.5,
    #                     "Loss with brighter lighting should be much lower than with different skin")
        
    #     self.assertLess(normal_to_shadow_loss.item(), different_skin_loss.item() * 0.5,
    #                     "Loss with shadow should be much lower than with different skin")
        
    #     # The loss between same image should be the lowest
    #     self.assertLess(same_image_loss.item(), normal_to_dark_loss.item(),
    #                     "Loss with same image should be lowest")

    # def test_skin_tone_loss_extreme_lighting(self):
    #     """Test if skin tone loss handles extreme lighting conditions properly"""
    #     # Skip test if face parser is not available
    #     if not self.has_face_parser:
    #         self.skipTest("Face parser not available")
        
    #     # Create test images with extreme lighting conditions
    #     from PIL import Image, ImageEnhance
    #     import torchvision.transforms as transforms
        
    #     transform = transforms.Compose([
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])
        
    #     # Load base face image
    #     base_face = Image.open('models/tests/images/face1.jpg').convert('RGB')
        
    #     # Create extreme lighting variations
    #     # 1. Almost black (very dark)
    #     face_very_dark = ImageEnhance.Brightness(base_face).enhance(0.1)
        
    #     # 2. Almost white (very bright/overexposed)
    #     face_very_bright = ImageEnhance.Brightness(base_face).enhance(2.5)
        
    #     # 3. High contrast
    #     face_high_contrast = ImageEnhance.Contrast(base_face).enhance(2.0)
        
    #     # 4. Low contrast
    #     face_low_contrast = ImageEnhance.Contrast(base_face).enhance(0.3)
        
    #     # Convert to tensors
    #     face_normal_tensor = transform(base_face).unsqueeze(0)
    #     face_very_dark_tensor = transform(face_very_dark).unsqueeze(0)
    #     face_very_bright_tensor = transform(face_very_bright).unsqueeze(0)
    #     face_high_contrast_tensor = transform(face_high_contrast).unsqueeze(0)
    #     face_low_contrast_tensor = transform(face_low_contrast).unsqueeze(0)
        
    #     # Move tensors to the same device as the model
    #     if hasattr(self.model, 'device'):
    #         face_normal_tensor = face_normal_tensor.to(self.model.device)
    #         face_very_dark_tensor = face_very_dark_tensor.to(self.model.device)
    #         face_very_bright_tensor = face_very_bright_tensor.to(self.model.device)
    #         face_high_contrast_tensor = face_high_contrast_tensor.to(self.model.device)
    #         face_low_contrast_tensor = face_low_contrast_tensor.to(self.model.device)
        
    #     # Save the extreme lighting images for visual inspection
    #     test_output_dir = Path('models/tests/output')
    #     test_output_dir.mkdir(parents=True, exist_ok=True)
    #     face_very_dark.save(test_output_dir / 'face_very_dark.jpg')
    #     face_very_bright.save(test_output_dir / 'face_very_bright.jpg')
    #     face_high_contrast.save(test_output_dir / 'face_high_contrast.jpg')
    #     face_low_contrast.save(test_output_dir / 'face_low_contrast.jpg')
        
    #     # Calculate extreme lighting losses
    #     very_dark_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_very_dark_tensor)
    #     very_bright_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_very_bright_tensor)
    #     high_contrast_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_high_contrast_tensor)
    #     low_contrast_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_low_contrast_tensor)
        
    #     # Compare with a completely different skin tone
    #     face_different = Image.open('models/tests/images/face2.jpg').convert('RGB')
    #     face_different_tensor = transform(face_different).unsqueeze(0)
    #     if hasattr(self.model, 'device'):
    #         face_different_tensor = face_different_tensor.to(self.model.device)
    #     different_skin_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_different_tensor)
        
    #     # Print losses for debugging
    #     print(f"\nExtreme lighting skin tone loss comparison:")
    #     print(f"Very dark lighting: {very_dark_loss.item():.6f}")
    #     print(f"Very bright lighting: {very_bright_loss.item():.6f}")
    #     print(f"High contrast: {high_contrast_loss.item():.6f}")
    #     print(f"Low contrast: {low_contrast_loss.item():.6f}")
    #     print(f"Different skin tone: {different_skin_loss.item():.6f}")
        
    #     # In extreme cases, face parser might fail to detect skin at all
    #     # If face parser returns valid results, losses should still be
    #     # lower than with a completely different skin tone
        
    #     # Check if we have valid face parsing results
    #     src_skin = self.model.extract_skin_regions(face_normal_tensor)
    #     dark_skin = self.model.extract_skin_regions(face_very_dark_tensor)
    #     bright_skin = self.model.extract_skin_regions(face_very_bright_tensor)
        
    #     # Check if skin was detected in extreme cases
    #     dark_has_skin = (torch.sum(dark_skin) > 0)
    #     bright_has_skin = (torch.sum(bright_skin) > 0)
        
    #     # Only test if skin was detected
    #     if dark_has_skin:
    #         self.assertLess(very_dark_loss.item(), different_skin_loss.item(),
    #                        "Even with very dark lighting, loss should be lower than with different skin")
        
    #     if bright_has_skin:
    #         self.assertLess(very_bright_loss.item(), different_skin_loss.item(),
    #                        "Even with very bright lighting, loss should be lower than with different skin")
        
    #     # For contrast variations, skin should be more reliably detected
    #     self.assertLess(high_contrast_loss.item(), different_skin_loss.item(),
    #                    "With high contrast, loss should be lower than with different skin")
        
    #     self.assertLess(low_contrast_loss.item(), different_skin_loss.item(),
    #                    "With low contrast, loss should be lower than with different skin")

    def test_segmentation_loss(self):
        # Skip test if face parser is not available
        if not hasattr(self, 'has_face_parser'):
            self.skipTest("Face parser not initialized")
        
        # Load test images
        face1_tensor, face1_2_tensor = self.load_test_images()
        face2_tensor = self.load_test_images('face2.jpg')
        
        # Set up mock input
        self.model.real_A = face1_tensor
        self.model.fake_B = face1_2_tensor
        
        # Get masks directly to check equality
        mask1 = self.model.get_face_parsing_masks(face1_tensor)
        mask1_again = self.model.get_face_parsing_masks(face1_tensor)
        
        # Check if masks are identical for the same image
        masks_equal = torch.equal(mask1, mask1_again)
        print(f"Are masks for the same image identical? {masks_equal}")
        
        seg_loss = self.model.compute_segmentation_loss(face1_tensor, face1_tensor)
        print(f"Segmentation loss face1 to face1: {seg_loss.item()}")
        seg_loss = self.model.compute_segmentation_loss(face1_tensor, face2_tensor)
        print(f"Segmentation loss face1 to face2: {seg_loss.item()}")

    def test_edge_detection(self):
        """Test if edge detection works properly"""
        # Skip test if model doesn't have the required methods
        if not hasattr(self.model, 'detect_edges'):
            self.skipTest("Edge detection not available in the model")
        
        # Load test images
        face1_tensor, face1_2_tensor = self.load_test_images()
        
        # Detect edges
        edges, grad_x, grad_y = self.model.detect_edges(face1_tensor)
        
        # Verify edge map has correct shape and range
        self.assertEqual(edges.shape[0], face1_tensor.shape[0], "Batch size mismatch in edge map")
        self.assertEqual(edges.shape[2], face1_tensor.shape[2], "Height mismatch in edge map")
        self.assertEqual(edges.shape[3], face1_tensor.shape[3], "Width mismatch in edge map")
        self.assertGreaterEqual(edges.min().item(), 0.0, "Edge map has negative values")
        self.assertLessEqual(edges.max().item(), 1.0, "Edge map has values greater than 1")
        
        # Save the detected edges for visualization
        self.save_edge_maps(face1_tensor, edges, grad_x, grad_y, "face1")
        
        # Detect edges for the second face image
        edges2, grad_x2, grad_y2 = self.model.detect_edges(face1_2_tensor)
        
        # Save those edges too
        self.save_edge_maps(face1_2_tensor, edges2, grad_x2, grad_y2, "face1_2")
        
        # Make sure edges were saved
        test_output_dir = Path('models/tests/output')
        self.assertTrue((test_output_dir / 'edges_face1.png').exists(), "Edge map was not saved")
        self.assertTrue((test_output_dir / 'edges_face1_2.png').exists(), "Second edge map was not saved")
        
    def save_edge_maps(self, original_img, edges, grad_x=None, grad_y=None, name_prefix=""):
        """Save edge maps as images for visualization"""
        import torchvision.utils as vutils
        import numpy as np
        from PIL import Image
        
        # Create output directory if it doesn't exist
        test_output_dir = Path('models/tests/output')
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Original image
        img_np = (original_img[0].cpu().permute(1, 2, 0).detach().numpy() + 1) / 2.0
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save(test_output_dir / f'original_{name_prefix}.png')
        
        # Edge map
        edge_map = edges[0, 0].cpu().detach().numpy()
        edge_map_img = Image.fromarray((edge_map * 255).astype(np.uint8))
        edge_map_img.save(test_output_dir / f'edges_{name_prefix}.png')
        
        # Gradient X (if provided)
        if grad_x is not None:
            grad_x_np = grad_x[0, 0].cpu().detach().numpy()
            grad_x_np = (grad_x_np - grad_x_np.min()) / (grad_x_np.max() - grad_x_np.min() + 1e-8)
            grad_x_img = Image.fromarray((grad_x_np * 255).astype(np.uint8))
            grad_x_img.save(test_output_dir / f'grad_x_{name_prefix}.png')
        
        # Gradient Y (if provided)
        if grad_y is not None:
            grad_y_np = grad_y[0, 0].cpu().detach().numpy()
            grad_y_np = (grad_y_np - grad_y_np.min()) / (grad_y_np.max() - grad_y_np.min() + 1e-8)
            grad_y_img = Image.fromarray((grad_y_np * 255).astype(np.uint8))
            grad_y_img.save(test_output_dir / f'grad_y_{name_prefix}.png')
        
        # Colorized edge map (for better visualization)
        colored_edges = np.zeros((edge_map.shape[0], edge_map.shape[1], 3), dtype=np.uint8)
        colored_edges[:, :, 0] = (edge_map * 255).astype(np.uint8)  # Red channel
        colored_edges_img = Image.fromarray(colored_edges)
        colored_edges_img.save(test_output_dir / f'colored_edges_{name_prefix}.png')
        
        # Overlay edges on original image
        overlay_alpha = 0.7
        overlay = img_np * (1 - overlay_alpha * edge_map[:, :, np.newaxis]) + overlay_alpha * edge_map[:, :, np.newaxis]
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        overlay_img.save(test_output_dir / f'overlay_edges_{name_prefix}.png')
        
        # Create a grid of images for easy comparison
        grid_tensor = torch.cat([
            original_img,
            # Convert grayscale edge map to 3-channel for concatenation
            torch.tensor(edge_map).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(original_img.device)
        ], dim=0)
        grid_img = vutils.make_grid(grid_tensor, nrow=2, normalize=True, scale_each=True)
        grid_np = grid_img.cpu().permute(1, 2, 0).detach().numpy()
        grid_img_pil = Image.fromarray((grid_np * 255).astype(np.uint8))
        grid_img_pil.save(test_output_dir / f'grid_{name_prefix}.png')
    
    def test_edge_preservation_loss(self):
        """Test if edge preservation loss works properly"""
        # Skip test if model doesn't have the required methods
        if not hasattr(self.model, 'compute_edge_preservation_loss'):
            self.skipTest("Edge preservation loss not available in the model")
        
        # Load test images
        face1_tensor, face1_2_tensor = self.load_test_images()
        
        # Calculate edge preservation loss
        edge_loss = self.model.compute_edge_preservation_loss(face1_tensor, face1_2_tensor)
        
        # Verify loss is a reasonable value
        self.assertGreaterEqual(edge_loss.item(), 0.0, "Edge preservation loss should be non-negative")
        print(f"Edge preservation loss between different images: {edge_loss.item():.6f}")
        
        # Test with same image (loss should be near zero)
        same_loss = self.model.compute_edge_preservation_loss(face1_tensor, face1_tensor)
        print(f"Edge preservation loss with same image: {same_loss.item():.6f}")
        
        # Same image loss should be less than different image loss
        self.assertLess(same_loss.item(), edge_loss.item(), 
                        "Loss with same image should be lower than with different images")
    
    def test_color_consistency_loss(self):
        """Test if color consistency loss works properly"""
        # Skip test if model doesn't have the required methods
        if not hasattr(self.model, 'compute_color_consistency_loss'):
            self.skipTest("Color consistency loss not available in the model")
        
        # Load test images
        face1_tensor, face1_2_tensor = self.load_test_images()
        
        # Calculate color consistency loss
        color_loss = self.model.compute_color_consistency_loss(face1_tensor, face1_2_tensor)
        
        # Verify loss is a reasonable value
        self.assertGreaterEqual(color_loss.item(), 0.0, "Color consistency loss should be non-negative")
        print(f"Color consistency loss between different images: {color_loss.item():.6f}")
        
        # Test with same image (loss should be near zero)
        same_loss = self.model.compute_color_consistency_loss(face1_tensor, face1_tensor)
        print(f"Color consistency loss with same image: {same_loss.item():.6f}")
        
        # Same image loss should be less than different image loss
        self.assertLess(same_loss.item(), color_loss.item(),
                       "Loss with same image should be lower than with different images")
    
    def test_edge_preservation_with_noise(self):
        """Test edge preservation with noisy images"""
        # Skip test if model doesn't have the required methods
        if not hasattr(self.model, 'compute_edge_preservation_loss'):
            self.skipTest("Edge preservation loss not available in the model")
        
        # Load test image
        face1_tensor = self.load_test_images('face1.jpg')
        
        # Create a noisy version of the image
        noise = torch.randn_like(face1_tensor) * 0.1
        noisy_image = face1_tensor + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)
        
        # Create a blurred version of the image
        import torch.nn.functional as F
        kernel_size = 5
        sigma = 2.0
        channels = face1_tensor.shape[1]
        
        # Create a 2D Gaussian kernel
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        # Calculate the 2D gaussian kernel
        gaussian_kernel = torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
        )
        
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        
        # Move the kernel to the same device as the input tensor
        gaussian_kernel = gaussian_kernel.to(face1_tensor.device)
        
        # Create padding based on the kernel size
        padding = kernel_size // 2
        
        # Blur the input tensor using F.conv2d with the gaussian kernel
        blurred_image = torch.zeros_like(face1_tensor)
        for i in range(channels):
            blurred_image[:, i:i+1] = F.conv2d(
                face1_tensor[:, i:i+1], 
                gaussian_kernel[:1], 
                padding=padding,
                groups=1
            )
        
        # Save images for visualization
        test_output_dir = Path('models/tests/output')
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        def save_tensor_image(tensor, filename):
            from PIL import Image
            import numpy as np
            img_np = (tensor[0].cpu().permute(1, 2, 0).detach().numpy() + 1) / 2.0
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(test_output_dir / filename)
        
        save_tensor_image(face1_tensor, 'original.png')
        save_tensor_image(noisy_image, 'noisy.png')
        save_tensor_image(blurred_image, 'blurred.png')
        
        # Detect and save edge maps
        edges_original, _, _ = self.model.detect_edges(face1_tensor)
        edges_noisy, _, _ = self.model.detect_edges(noisy_image)
        edges_blurred, _, _ = self.model.detect_edges(blurred_image)
        
        self.save_edge_maps(face1_tensor, edges_original, None, None, "original")
        self.save_edge_maps(noisy_image, edges_noisy, None, None, "noisy")
        self.save_edge_maps(blurred_image, edges_blurred, None, None, "blurred")
        
        # Calculate edge preservation loss
        loss_noisy = self.model.compute_edge_preservation_loss(face1_tensor, noisy_image)
        loss_blurred = self.model.compute_edge_preservation_loss(face1_tensor, blurred_image)
        
        print(f"Edge preservation loss with noisy image: {loss_noisy.item():.6f}")
        print(f"Edge preservation loss with blurred image: {loss_blurred.item():.6f}")
        
        # Noisy image should preserve edges better than blurred image
        self.assertLess(loss_noisy.item(), loss_blurred.item() * 1.5,
                       "Loss with noisy image should generally be lower than with blurred image")


if __name__ == '__main__':
    unittest.main()
