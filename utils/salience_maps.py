import numpy as np
import torch.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, image, target_class=None, device='cuda'):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input tensor [1, C, H, W]
            target_class: Class to visualize (None = predicted)
        
        Returns:
            heatmap: Numpy array [H, W]
        """
        self.model.eval()
        image = image.to(device)
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class
    
def make_salience_map(model, image_path):
    target_layer = model.layer8 # last layer for the MergerCNN architecture. MAKE FUNCTIONALITY TO INPUT LAYER LATER
    gradcam = GradCAM(model, target_layer)
    image_tensor = torch.load(f"{image_path}.pt")
    
    salience_map, pred_class = gradcam(image_tensor)
    
    return salience_map, pred_class