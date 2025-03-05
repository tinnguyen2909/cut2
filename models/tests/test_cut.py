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
        self.gpu_ids = []
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
        face2_path = dir_path / 'face2.jpg'
        
        if not face1_path.exists() or not face2_path.exists():
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
            img2.save(face2_path)

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
        face2 = Image.open('models/tests/images/face2.jpg').convert('RGB')
        
        face1_tensor = transform(face1).unsqueeze(0)
        face2_tensor = transform(face2).unsqueeze(0)
        
        return face1_tensor, face2_tensor

    def test_eye_color_loss(self):
        """Test if eye color loss works properly"""
        # Skip test if face parser is not available
        if not hasattr(self, 'has_face_parser'):
            self.skipTest("Face parser not initialized")
        
        # Load test images
        face1_tensor, face2_tensor = self.load_test_images()
        
        # Set up mock input
        self.model.real_A = face1_tensor
        self.model.fake_B = face2_tensor
        
        # Calculate eye color loss
        eye_loss = self.model.compute_eye_color_loss(face1_tensor, face2_tensor)
        
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
        face1_tensor, face2_tensor = self.load_test_images()
        
        # Set up mock input
        self.model.real_A = face1_tensor
        self.model.fake_B = face2_tensor
        
        # Calculate skin tone loss
        skin_loss = self.model.compute_skin_tone_loss(face1_tensor, face2_tensor)
        
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
        masks2 = self.model.get_face_parsing_masks(face1_tensor)
        second_call_time = time.time() - start_time
        
        # Second call should be significantly faster
        self.assertLess(second_call_time, first_call_time * 0.5, 
                        f"Cached call ({second_call_time:.4f}s) not faster than original call ({first_call_time:.4f}s)")
        
        # Results should be identical
        self.assertTrue(torch.equal(masks1, masks2))
        
        # Cache should have exactly one entry
        self.assertEqual(len(self.model.face_parsing_cache), 1)

    # def test_combined_losses_in_G_loss(self):
    #     """Test if eye and skin losses are included in G_loss calculation"""
    #     # Skip test if face parser is not available
    #     if not hasattr(self, 'has_face_parser'):
    #         self.skipTest("Face parser not initialized")
        
    #     # Load test images
    #     face1_tensor, face2_tensor = self.load_test_images()
        
    #     # Set up mock input
    #     self.model.real_A = face1_tensor
    #     self.model.fake_B = face2_tensor
        
    #     # Mock other required attributes to compute G_loss
    #     self.model.loss_G_GAN = torch.tensor(1.0)
    #     self.model.loss_NCE = torch.tensor(1.0)
        
    #     # Calculate individual losses
    #     eye_loss = self.model.compute_eye_color_loss(face1_tensor, face2_tensor) * self.model.opt.lambda_eye
    #     skin_loss = self.model.compute_skin_tone_loss(face1_tensor, face2_tensor) * self.model.opt.lambda_skin
        
    #     # Assign losses to model
    #     self.model.loss_eye = eye_loss
    #     self.model.loss_skin = skin_loss
        
    #     # Expected total loss
    #     expected_loss = self.model.loss_G_GAN + self.model.loss_NCE + eye_loss + skin_loss
        
    #     # Actual total loss - need to patch compute_G_loss to avoid 
    #     # recalculating the individual losses
    #     original_compute_eye = self.model.compute_eye_color_loss
    #     original_compute_skin = self.model.compute_skin_tone_loss
        
    #     self.model.compute_eye_color_loss = lambda x, y: eye_loss / self.model.opt.lambda_eye
    #     self.model.compute_skin_tone_loss = lambda x, y: skin_loss / self.model.opt.lambda_skin
        
    #     # Calculate total G loss
    #     total_loss = self.model.compute_G_loss()
        
    #     # Restore original methods
    #     self.model.compute_eye_color_loss = original_compute_eye
    #     self.model.compute_skin_tone_loss = original_compute_skin
        
    #     # Total loss should match expected loss
    #     self.assertAlmostEqual(total_loss.item(), expected_loss.item(), places=5)


if __name__ == '__main__':
    unittest.main()
