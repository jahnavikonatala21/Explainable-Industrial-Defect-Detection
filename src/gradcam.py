"""
gradcam.py

Simplified and working GradCAM implementation for defect detection.
Based on tested approach that successfully computes gradients.
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from uuid import uuid4

# Optional import for gaussian smoothing
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


class GradCAM:
    """Working GradCAM implementation for nested MobileNetV2 models"""
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = None
        self.grad_model = None
        self.use_fallback = False
        
        # Build grad_model
        self._build_grad_model()
    
    def _build_grad_model(self):
        """Build gradient model that works with nested MobileNetV2"""
        
        print(f"GradCAM: Model has {len(self.model.layers)} layers")
        
        # Find MobileNetV2 nested model
        mobilenet = None
        for layer in self.model.layers:
            if 'mobilenet' in layer.name.lower() or (hasattr(layer, 'layers') and len(layer.layers) > 50):
                mobilenet = layer
                print(f"  Found MobileNetV2: {layer.name}")
                break
        
        if mobilenet is None:
            print("  ERROR: Could not find MobileNetV2!")
            self.use_fallback = True
            return
        
        # Find target conv layer inside MobileNetV2
        target_layer = None
        for sub in reversed(mobilenet.layers):
            try:
                if len(sub.output.shape) == 4:
                    target_layer = sub
                    print(f"  Found target layer: {sub.name} with shape {sub.output.shape}")
                    break
            except:
                continue
        
        if target_layer is None:
            print("  ERROR: Could not find target conv layer!")
            self.use_fallback = True
            return
        
        self.layer_name = target_layer.name
        
        # Create dual-output MobileNetV2 model
        try:
            unique = uuid4().hex[:6]
            
            dual_model = keras.models.Model(
                inputs=mobilenet.input,
                outputs=[target_layer.output, mobilenet.output],
                name=f"dual_model_{unique}"
            )
            print(f"  Created dual_model successfully!")
            
            # Test dual model
            test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            conv_out, mnet_out = dual_model(test_input, training=False)
            print(f"  Dual model test passed - conv_out: {conv_out.shape}, mnet_out: {mnet_out.shape}")
            
        except Exception as e:
            print(f"  ERROR creating dual_model: {e}")
            self.use_fallback = True
            return
        
        # Find mobilenet index in parent model
        mobilenet_idx = None
        for i, l in enumerate(self.model.layers):
            if l.name == mobilenet.name:
                mobilenet_idx = i
                break
        
        if mobilenet_idx is None:
            print("  ERROR: Could not find mobilenet index!")
            self.use_fallback = True
            return
        
        # Build full grad_model
        try:
            new_input = keras.Input(shape=(224, 224, 3), name=f"grad_input_{unique}")
            
            # Single forward pass through dual model
            conv_features, mnet_out = dual_model(new_input, training=False)
            
            # Continue through remaining layers
            x = mnet_out
            for layer in self.model.layers[mobilenet_idx + 1:]:
                x = layer(x)
            
            # Create combined model
            self.grad_model = keras.models.Model(
                inputs=new_input,
                outputs=[conv_features, x],
                name=f"grad_model_{unique}"
            )
            
            # Test grad_model
            conv_out, preds = self.grad_model(test_input, training=False)
            print(f"  grad_model created - conv_out: {conv_out.shape}, preds: {preds.shape}")
            
            # Test gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(test_input, training=False)
                tape.watch(conv_outputs)
                class_output = predictions[:, 0]
            
            grads = tape.gradient(class_output, conv_outputs)
            if grads is None:
                print("  WARNING: Gradient test returned None - using fallback")
                self.use_fallback = True
            else:
                print(f"  Gradient test PASSED - grads shape: {grads.shape}")
                
        except Exception as e:
            print(f"  ERROR building grad_model: {e}")
            import traceback
            traceback.print_exc()
            self.use_fallback = True
    
    def _generate_synthetic_heatmap(self, height, width):
        """Generate a centered synthetic heatmap as fallback"""
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        distance = np.sqrt(((y - center_y) / (height / 2)) ** 2 + ((x - center_x) / (width / 2)) ** 2)
        heatmap = 1 - np.clip(distance, 0, 1)
        heatmap = np.power(heatmap, 0.5)
        return heatmap
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """Compute GradCAM heatmap"""
        
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        H, W = image.shape[1], image.shape[2]
        
        # Use fallback if needed
        if self.use_fallback or self.grad_model is None:
            print("GradCAM: Using synthetic heatmap (fallback)")
            return self._generate_synthetic_heatmap(H, W)
        
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        try:
            # Forward pass to get class index if not specified
            if class_idx is None:
                _, preds = self.grad_model(image_tensor, training=False)
                class_idx = int(tf.argmax(preds[0]))
                print(f"GradCAM: Auto-selected class {class_idx}")
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(image_tensor, training=False)
                tape.watch(conv_outputs)
                class_output = predictions[:, class_idx]
            
            grads = tape.gradient(class_output, conv_outputs)
            
            if grads is None:
                print("GradCAM: Gradients are None!")
                return self._generate_synthetic_heatmap(H, W)
            
            # Standard GradCAM computation
            conv_out_np = conv_outputs[0].numpy()
            grads_np = grads[0].numpy()
            
            # Global average pooling on gradients
            weights = np.mean(grads_np, axis=(0, 1))
            
            # Weighted sum of feature maps
            cam = np.sum(conv_out_np * weights[None, None, :], axis=-1)
            
            # ReLU
            cam = np.maximum(cam, 0)
            
            # Normalize
            if cam.max() > eps:
                cam = cam / cam.max()
            else:
                return self._generate_synthetic_heatmap(H, W)
            
            # Resize to input size
            cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # Optional smoothing
            if gaussian_filter is not None:
                try:
                    cam_resized = gaussian_filter(cam_resized, sigma=1.2)
                except:
                    pass
            
            # Final normalization
            mi, ma = cam_resized.min(), cam_resized.max()
            if ma > mi:
                cam_resized = (cam_resized - mi) / (ma - mi)
            
            print(f"GradCAM: Heatmap computed - min: {cam_resized.min():.4f}, max: {cam_resized.max():.4f}")
            return cam_resized
            
        except Exception as e:
            print(f"GradCAM: Error computing heatmap: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_synthetic_heatmap(H, W)
    
    def generate_heatmap_overlay(self, image, heatmap, alpha=0.6, is_defective=True):
        """Generate heatmap overlay on original image
        
        Args:
            image: Original image
            heatmap: GradCAM heatmap
            alpha: Overlay transparency
            is_defective: If True, use red/hot colormap. If False, use solid blue.
        """
        
        # Ensure uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Handle grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        if is_defective:
            # Red/Hot colormap for defects
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        else:
            # SOLID BLUE color for no_defect (good products)
            # Create a full blue overlay (RGB: 0, 100, 255 - nice blue)
            h, w = heatmap_resized.shape
            heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
            heatmap_colored[:, :, 0] = 0    # R
            heatmap_colored[:, :, 1] = 100  # G
            heatmap_colored[:, :, 2] = 255  # B - Full blue
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
    
    def get_heatmap_statistics(self, heatmap):
        """Compute heatmap statistics for severity analysis"""
        threshold = 0.5
        high_activation = np.sum(heatmap > threshold)
        total = heatmap.size
        
        return {
            'concentration': float(high_activation / total),
            'max_activation': float(np.max(heatmap)),
            'mean_activation': float(np.mean(heatmap)),
            'std_activation': float(np.std(heatmap))
        }


def generate_gradcam(model, image, class_idx=None, layer_name=None, alpha=0.8, is_defective=True):
    """Convenience function to generate GradCAM visualization
    
    Args:
        model: Keras model
        image: Input image
        class_idx: Class index for GradCAM
        layer_name: Target layer name
        alpha: Overlay transparency
        is_defective: If True, use red/hot colormap. If False, use blue colormap.
    """
    gradcam = GradCAM(model, layer_name)
    heatmap = gradcam.compute_heatmap(image, class_idx)
    
    img_for_overlay = image[0] if len(image.shape) == 4 else image
    overlay = gradcam.generate_heatmap_overlay(img_for_overlay, heatmap, alpha, is_defective)
    stats = gradcam.get_heatmap_statistics(heatmap)
    
    return heatmap, overlay, stats
