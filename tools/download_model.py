import torch
import torchvision.models as models

# Load the ResNet50 model with pretrained weights
model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
model.eval()

# Example input for tracing the model
example_input = torch.rand(1, 3, 224, 224)

# Trace the model to convert it to Torch Script
traced_model = torch.jit.trace(model, example_input)

# Save the traced model to a file
traced_model.save("resnet50_torch.pth")

print("ResNet50 model converted to Torch Script and saved as resnet50_torch.pth")