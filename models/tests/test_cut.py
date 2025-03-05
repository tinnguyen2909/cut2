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

    def load_test_images(self):
        """Load test images into tensors"""
        from PIL import Image
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        face1 = Image.open('models/tests/images/face1.jpg').convert('RGB')
        face1_2 = Image.open('models/tests/images/face1_2.jpg').convert('RGB')
        
        face1_tensor = transform(face1).unsqueeze(0)
        face1_2_tensor = transform(face1_2).unsqueeze(0)
        
        # Move tensors to the same device as the model
        if hasattr(self.model, 'device'):
            face1_tensor = face1_tensor.to(self.model.device)
            face1_2_tensor = face1_2_tensor.to(self.model.device)
        
        return face1_tensor, face1_2_tensor

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

    def test_skin_tone_loss_lighting_robustness(self):
        """Test if skin tone loss is robust against lighting variations"""
        # Skip test if face parser is not available
        if not self.has_face_parser:
            self.skipTest("Face parser not available")
        
        # Create test images with same skin tone but different lighting conditions
        from PIL import Image, ImageEnhance
        import torchvision.transforms as transforms
        import numpy as np
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load or create base face image
        base_face = Image.open('models/tests/images/face1.jpg').convert('RGB')
        
        # Create variations with different lighting conditions
        # 1. Normal lighting (original)
        face_normal = base_face.copy()
        
        # 2. Darker version (simulating low light)
        face_dark = ImageEnhance.Brightness(base_face).enhance(0.5)
        
        # 3. Brighter version (simulating strong light)
        face_bright = ImageEnhance.Brightness(base_face).enhance(1.5)
        
        # 4. Load shadow face image directly
        face_shadow = Image.open('models/tests/images/face1_shadow.jpg').convert('RGB')
        
        # 5. Different skin tone (for comparison)
        face_different = Image.open('models/tests/images/face2.jpg').convert('RGB')
        
        # Convert to tensors
        face_normal_tensor = transform(face_normal).unsqueeze(0)
        face_dark_tensor = transform(face_dark).unsqueeze(0)
        face_bright_tensor = transform(face_bright).unsqueeze(0)
        face_shadow_tensor = transform(face_shadow).unsqueeze(0)
        face_different_tensor = transform(face_different).unsqueeze(0)
        
        # Move tensors to the same device as the model
        if hasattr(self.model, 'device'):
            face_normal_tensor = face_normal_tensor.to(self.model.device)
            face_dark_tensor = face_dark_tensor.to(self.model.device)
            face_bright_tensor = face_bright_tensor.to(self.model.device)
            face_shadow_tensor = face_shadow_tensor.to(self.model.device)
            face_different_tensor = face_different_tensor.to(self.model.device)
        
        # Calculate skin tone losses between same skin under different lighting
        normal_to_dark_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_dark_tensor)
        normal_to_bright_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_bright_tensor)
        normal_to_shadow_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_shadow_tensor)
        
        # Calculate skin tone loss between different skin tones
        different_skin_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_different_tensor)
        
        # Calculate reference loss (same image, same lighting)
        same_image_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_normal_tensor)
        
        # Print losses for debugging
        print(f"\nSkin tone loss comparison:")
        print(f"Same image, same lighting: {same_image_loss.item():.6f}")
        print(f"Same skin, dark lighting: {normal_to_dark_loss.item():.6f}")
        print(f"Same skin, bright lighting: {normal_to_bright_loss.item():.6f}")
        print(f"Same skin, partial shadow: {normal_to_shadow_loss.item():.6f}")
        print(f"Different skin tone: {different_skin_loss.item():.6f}")
        
        # Losses between the same skin under different lighting should be
        # significantly lower than between different skin tones
        self.assertLess(normal_to_dark_loss.item(), different_skin_loss.item() * 0.5,
                        "Loss with darker lighting should be much lower than with different skin")
        
        self.assertLess(normal_to_bright_loss.item(), different_skin_loss.item() * 0.5,
                        "Loss with brighter lighting should be much lower than with different skin")
        
        self.assertLess(normal_to_shadow_loss.item(), different_skin_loss.item() * 0.5,
                        "Loss with shadow should be much lower than with different skin")
        
        # The loss between same image should be the lowest
        self.assertLess(same_image_loss.item(), normal_to_dark_loss.item(),
                        "Loss with same image should be lowest")

    def test_skin_tone_loss_extreme_lighting(self):
        """Test if skin tone loss handles extreme lighting conditions properly"""
        # Skip test if face parser is not available
        if not self.has_face_parser:
            self.skipTest("Face parser not available")
        
        # Create test images with extreme lighting conditions
        from PIL import Image, ImageEnhance
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load base face image
        base_face = Image.open('models/tests/images/face1.jpg').convert('RGB')
        
        # Create extreme lighting variations
        # 1. Almost black (very dark)
        face_very_dark = ImageEnhance.Brightness(base_face).enhance(0.1)
        
        # 2. Almost white (very bright/overexposed)
        face_very_bright = ImageEnhance.Brightness(base_face).enhance(2.5)
        
        # 3. High contrast
        face_high_contrast = ImageEnhance.Contrast(base_face).enhance(2.0)
        
        # 4. Low contrast
        face_low_contrast = ImageEnhance.Contrast(base_face).enhance(0.3)
        
        # Convert to tensors
        face_normal_tensor = transform(base_face).unsqueeze(0)
        face_very_dark_tensor = transform(face_very_dark).unsqueeze(0)
        face_very_bright_tensor = transform(face_very_bright).unsqueeze(0)
        face_high_contrast_tensor = transform(face_high_contrast).unsqueeze(0)
        face_low_contrast_tensor = transform(face_low_contrast).unsqueeze(0)
        
        # Move tensors to the same device as the model
        if hasattr(self.model, 'device'):
            face_normal_tensor = face_normal_tensor.to(self.model.device)
            face_very_dark_tensor = face_very_dark_tensor.to(self.model.device)
            face_very_bright_tensor = face_very_bright_tensor.to(self.model.device)
            face_high_contrast_tensor = face_high_contrast_tensor.to(self.model.device)
            face_low_contrast_tensor = face_low_contrast_tensor.to(self.model.device)
        
        # Save the extreme lighting images for visual inspection
        test_output_dir = Path('models/tests/output')
        test_output_dir.mkdir(parents=True, exist_ok=True)
        face_very_dark.save(test_output_dir / 'face_very_dark.jpg')
        face_very_bright.save(test_output_dir / 'face_very_bright.jpg')
        face_high_contrast.save(test_output_dir / 'face_high_contrast.jpg')
        face_low_contrast.save(test_output_dir / 'face_low_contrast.jpg')
        
        # Calculate extreme lighting losses
        very_dark_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_very_dark_tensor)
        very_bright_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_very_bright_tensor)
        high_contrast_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_high_contrast_tensor)
        low_contrast_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_low_contrast_tensor)
        
        # Compare with a completely different skin tone
        face_different = Image.open('models/tests/images/face2.jpg').convert('RGB')
        face_different_tensor = transform(face_different).unsqueeze(0)
        if hasattr(self.model, 'device'):
            face_different_tensor = face_different_tensor.to(self.model.device)
        different_skin_loss = self.model.compute_skin_tone_loss(face_normal_tensor, face_different_tensor)
        
        # Print losses for debugging
        print(f"\nExtreme lighting skin tone loss comparison:")
        print(f"Very dark lighting: {very_dark_loss.item():.6f}")
        print(f"Very bright lighting: {very_bright_loss.item():.6f}")
        print(f"High contrast: {high_contrast_loss.item():.6f}")
        print(f"Low contrast: {low_contrast_loss.item():.6f}")
        print(f"Different skin tone: {different_skin_loss.item():.6f}")
        
        # In extreme cases, face parser might fail to detect skin at all
        # If face parser returns valid results, losses should still be
        # lower than with a completely different skin tone
        
        # Check if we have valid face parsing results
        src_skin = self.model.extract_skin_regions(face_normal_tensor)
        dark_skin = self.model.extract_skin_regions(face_very_dark_tensor)
        bright_skin = self.model.extract_skin_regions(face_very_bright_tensor)
        
        # Check if skin was detected in extreme cases
        dark_has_skin = (torch.sum(dark_skin) > 0)
        bright_has_skin = (torch.sum(bright_skin) > 0)
        
        # Only test if skin was detected
        if dark_has_skin:
            self.assertLess(very_dark_loss.item(), different_skin_loss.item(),
                           "Even with very dark lighting, loss should be lower than with different skin")
        
        if bright_has_skin:
            self.assertLess(very_bright_loss.item(), different_skin_loss.item(),
                           "Even with very bright lighting, loss should be lower than with different skin")
        
        # For contrast variations, skin should be more reliably detected
        self.assertLess(high_contrast_loss.item(), different_skin_loss.item(),
                       "With high contrast, loss should be lower than with different skin")
        
        self.assertLess(low_contrast_loss.item(), different_skin_loss.item(),
                       "With low contrast, loss should be lower than with different skin")


if __name__ == '__main__':
    unittest.main()
