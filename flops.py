import torch
from ptflops import get_model_complexity_info
import time

def calculate_flops_and_time(model, input_tensor):
    # Move the model to the appropriate device
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()  # Ensure the model is in evaluation mode
    
    # Calculate FLOPs and parameter count
    flops, params = get_model_complexity_info(
        model, 
        (3, 224, 224),  # Input size (channels, height, width)
        as_strings=True, 
        print_per_layer_stat=False
    )
    
    # Measure inference time
    input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_tensor)
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return flops, params, elapsed_time
