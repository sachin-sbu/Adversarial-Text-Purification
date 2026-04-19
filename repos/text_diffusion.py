import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from noise_lib import LogLinearNoise
from graph_lib import Absorbing
from sampling import get_sampling_fn
from model import SEDD
from load_model import load_model
from omegaconf import OmegaConf
from model import utils as mutils

class TextDiffusion:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Load the SEDD model, graph, and noise
        self.score_model, self.graph, self.noise = load_model(model_path, device)
        self.score_model.eval()
        
        # Initialize tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Initialize sampling function
        self.sampling_fn = get_sampling_fn(
            self.score_model.config,
            self.graph,
            self.noise,
            (1, self.score_model.config.model.length),
            1e-5,
            device
        )
        
    def add_noise(self, text, timestep):
        """
        Add noise to text by masking tokens based on timestep
        Args:
            text: Input text string
            timestep: Float between 0 and 1, higher means more noise
        Returns:
            Noised text with some tokens masked
        """
        # Tokenize input text
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Get noise level for this timestep
        sigma = self.noise.total_noise(torch.tensor([timestep], device=self.device))
        
        # Sample transitions based on noise level
        noised_tokens = self.graph.sample_transition(tokens, sigma)
        
        # Convert back to text, replacing masked tokens with [MASK]
        noised_text = self.tokenizer.decode(noised_tokens[0])
        return noised_text
    
    def denoise(self, noised_text, timestep):
        """
        Denoise text using the SEDD model
        Args:
            noised_text: Text with masked tokens
            timestep: Float between 0 and 1, matching the timestep used for noising
        Returns:
            Denoised text
        """
        # Tokenize noised text
        tokens = self.tokenizer.encode(noised_text, return_tensors="pt").to(self.device)
        
        # Use the sampling function to denoise
        denoised_tokens = self.sampling_fn(self.score_model)
        
        # Convert back to text
        denoised_text = self.tokenizer.decode(denoised_tokens[0])
        return denoised_text

def main():
    # Example usage
    model_path = "/home/sdanisetty/projects/advpur/repos/Score-Entropy-Discrete-Diffusion/exp_local/fancyzhx/ag_news/2025.04.03/101142/"  # Replace with actual model path
    diffusion = TextDiffusion(model_path)
    
    # Original text
    text = "I possess the good common logical sense to realize that oil can not last forever and I am acutely aware of how much of my life in the suburbs spins around petrochemical products. I've been an avid consumer of new technology and I keep running out of space on powerboards"
    
    # Add noise with different timesteps
    timesteps = [0.2, 0.5, 0.8]
    for t in timesteps:
        noised = diffusion.add_noise(text, t)
        denoised = diffusion.denoise(noised, t)
        
        print(f"\nTimestep: {t}")
        print(f"Original: {text}")
        print(f"Noised: {noised}")
        print(f"Denoised: {denoised}")

if __name__ == "__main__":
    main() 