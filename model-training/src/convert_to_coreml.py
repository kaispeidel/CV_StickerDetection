import torch
import coremltools as ct
import segmentation_models_pytorch as smp

def convert_to_coreml():
    # Load your trained PyTorch model
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,  # We'll load our trained weights
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    
    # Load trained weights
    model.load_state_dict(torch.load("models/sticker_segmentation.pth"))
    model.eval()
    
    # Create example input
    example_input = torch.rand(1, 3, 256, 256)  # Batch, Channels, Height, Width
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input_image", shape=example_input.shape)],
        outputs=[ct.ImageType(name="segmentation_mask")],
        minimum_deployment_target=ct.target.iOS14
    )
    
    # Save the Core ML model
    coreml_model.save("models/StickerSegmentation.mlmodel")
    print("Core ML model saved!")

if __name__ == "__main__":
    convert_to_coreml()