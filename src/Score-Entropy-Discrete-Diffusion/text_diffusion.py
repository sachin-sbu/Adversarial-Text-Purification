import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from noise_lib import LogLinearNoise
from graph_lib import Absorbing
from sampling import Denoiser, get_predictor
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
        
        # Initialize denoiser
        self.denoiser = Denoiser(self.graph, self.noise)
        
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
        return noised_text, noised_tokens
    
    def denoise(self, noised_tokens, timestep, steps=10):
        """
        Denoise text using the SEDD model
        Args:
            noised_text: Text with masked tokens
            timestep: Float between 0 and 1, matching the timestep used for noising
            steps: Number of denoising steps
        Returns:
            Denoised text
        """
        # Tokenize noised text
        tokens = noised_tokens
        # x = self.graph.sample_limit(tokens.shape[0], tokens.shape[1]).to(self.device)
        x = noised_tokens
        
        # Initialize predictor
        predictor = get_predictor('analytic')(self.graph, self.noise)
        denoiser = Denoiser(self.graph, self.noise)
        # Get model predictions
        sampling_score_fn = mutils.get_score_fn(self.score_model, train=False, sampling=True)
        
        # Set up timesteps for denoising
        eps = 1e-5
        timesteps = torch.linspace(1, eps, steps + 1, device=self.device)
        dt = (1 - eps) / steps
        
        
        # Denoising loop
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)

            sentences = self.tokenizer.batch_decode(x)
            # import ipdb; ipdb.set_trace()
            # print(x)
            # if i%10 == 0 or i>90:
            #     print(t, ": " + sentences[0])
            #     print("--------------------------------")
        
        # Final denoising step
        t = 0.2 * torch.ones(tokens.shape[0], 1, device=self.device)
        denoised_tokens = self.denoiser.update_fn(sampling_score_fn, tokens, t)
        
        # Convert back to text
        denoised_text = self.tokenizer.decode(denoised_tokens[0])
        return denoised_text

def main():
    # Example usage
    model_path = "/home/sdanisetty/projects/advpur/repos/Score-Entropy-Discrete-Diffusion/exp_local/fancyzhx/ag_news/2025.04.03/101142/"  # Replace with actual model path
    diffusion = TextDiffusion(model_path)
    
    # Original text
    # og_text = "E-mail scam targets police chief Wiltshire Police warns about “phishing” after its fraud squad chief was targeted."
    # text = "E-mail scam targets gendarmerie chief Wiltshire Police warns about “phishing” after its deception battalion massa was targeted."

    og_text = "aussie ruling party leads in election polls , but gap narrows the government of prime minister john howard had a narrow lead in opinion polls heading into the final week of campaigning ahead of the australian federal election , but the opposition labor party was narrowing the gap , according"
    text = 	"aussie ruling party leads in election polls , but gap narrows the gov of prime minister john howard had a ltd culminate in opinion polls heading into the final week of campaigning ahead of the aus federal election , but the opposition labor party was downsize the disparity , according"

    # og_text = "I have the good common logical sense to know that oil can not last forever and I am acutely aware of how much of my life in the suburbs revolves around petrochemical products. I’ve been an avid consumer of new technology and I keep running out of space on powerboards - so..."
    # text = "I possess the good common logical sense to realize that oil can not last forever and I am acutely aware of how much of my life in the suburbs spins around petrochemical products. I’ve been an avid consumer of new technology and I keep running out of space on powerboards - well..."
    
    # Add noise with different timesteps
    timesteps = [0.1, 0.2, 0.3]
    for t in timesteps:
        noised, noisy_tokens = diffusion.add_noise(text, t)
        denoised = diffusion.denoise(noisy_tokens, t)
        print(f"\nTimestep: {t}")
        print(f"Original: {og_text}")
        print()
        print(f"Adversarial: {text}")
        print()
        print(f"Noised: {noised}")
        print()
        print()
        
        print(f"Denoised: {denoised}")
        print()

if __name__ == "__main__":
    main() 