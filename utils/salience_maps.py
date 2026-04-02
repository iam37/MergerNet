import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch

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
        #print(np.shape(image))
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
        cam = F.leaky_relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class
    
def make_salience_map(model, image_path):
    target_layer = model.layer1 # last layer for the MergerCNN architecture. MAKE FUNCTIONALITY TO INPUT LAYER LATER
    gradcam = GradCAM(model, target_layer)
    image_tensor = torch.load(f"{image_path}.pt")
    image = torch.load(f"{image_path}.pt", map_location='cpu')
    
    #print(f"Loaded shape: {image.shape}")
    
    if image.ndim == 3:
        # Check if it's [H, W, C] or [C, H, W]
        if image.shape[2] == 3 or image.shape[2] == 1:  # Likely [H, W, C]
            image = image.permute(2, 0, 1)  # → [C, H, W]
        # else: already [C, H, W]
    elif image.ndim == 2:
        # Single channel [H, W] → [1, H, W]
        image = image.unsqueeze(0)
    
    if image.ndim == 3:
        image = image.unsqueeze(0)
    
    #print(f"Fixed shape: {image.shape}")
    
    salience_map, pred_class = gradcam(image)
    
    return salience_map, pred_class