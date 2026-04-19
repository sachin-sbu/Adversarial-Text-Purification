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
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
import sys
sys.path.append("/home/sdanisetty/projects/advpur/repos/TextFooler/BERT/")
from run_classifier import convert_examples_to_features
from run_classifier import AGProcessor
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import numpy as np
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
output_config_file = "/home/sdanisetty/projects/advpur/repos/imdb/bert_config.json"  # Replace with your config file path
output_model_file = "/home/sdanisetty/projects/advpur/repos/imdb/pytorch_model.bin"  # Replace with your model file path
config = BertConfig(output_config_file)
model = BertForSequenceClassification(config, num_labels=2)
model.load_state_dict(torch.load(output_model_file))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
class_names = {0:"Negative", 1:"Positive"}


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
    model_path = "/home/sdanisetty/projects/advpur/repos/Score-Entropy-Discrete-Diffusion/exp_local/stanfordnlp/imdb/2025.04.23/140713/"  # Replace with actual model path
    diffusion = TextDiffusion(model_path)
    model.cuda()
    model.eval()
    

    fp = open("/home/sdanisetty/projects/advpur/repos/imdb_bert", "r")
    i = 0
    og_texts = []
    adv_texts = []
    for line in fp:
        # if i == 24:
        #     break
        if i%3 == 0:
            og_texts.append(line[15:-1])
            # print("-------------------")
        elif i%3 == 1:
            adv_texts.append(line[14:-1])
            # print('=====================')
        i += 1
    fp.close()

    accuracy = 0
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    acc_12 = 0
    acc_13 = 0
    acc_23 = 0
    pbar = tqdm(range(len(og_texts)), desc="Processing", dynamic_ncols=True)
    for i in pbar:
        og_text = og_texts[i][:512]
        adv_text = adv_texts[i][:512]
        tokens = tokenizer.tokenize(og_text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        max_seq_length = 128
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        out = model(torch.tensor([input_ids], device='cuda'), torch.tensor([segment_ids], device='cuda'), torch.tensor([input_mask], device='cuda'))
        # print('og_text :', class_names[np.argmax(out.detach().cpu().numpy(), axis=1)[0]])
        # print(out)
        og_label = class_names[np.argmax(out.detach().cpu().numpy(), axis=1)[0]]
        # print('og_text :', og_text)
        # print('og_label :', og_label)
        # print()

        tokens = tokenizer.tokenize(adv_text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        max_seq_length = 128
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        out = model(torch.tensor([input_ids], device='cuda'), torch.tensor([segment_ids], device='cuda'), torch.tensor([input_mask], device='cuda'))
        # print('adv_text :',class_names[np.argmax(out.detach().cpu().numpy(), axis=1)[0]])
        adv_label = class_names[np.argmax(out.detach().cpu().numpy(), axis=1)[0]]
        # print('adv_text :', adv_text)
        # print('adv_label :', adv_label)
        # print()
        
        timesteps = [0.1, 0.2, 0.3]
        score = 0
        scores = {0.1:0, 0.2:0, 0.3:0}
        for t in timesteps:
            noised, noisy_tokens = diffusion.add_noise(og_text, t)
            denoised = diffusion.denoise(noisy_tokens, t)
            # print(f"Denoised: {denoised}")
            # print()
            tokens = tokenizer.tokenize(denoised)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            max_seq_length = 128
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            out = model(torch.tensor([input_ids], device='cuda'), torch.tensor([segment_ids], device='cuda'), torch.tensor([input_mask], device='cuda'))
            # print('purified_text at ' + str(t) + " :",class_names[np.argmax(out.detach().cpu().numpy(), axis=1)[0]])
            label = class_names[np.argmax(out.detach().cpu().numpy(), axis=1)[0]]
            # print('purified_text at ' + str(t) + " :", denoised)
            # print('purified_label at ' + str(t) + " :", label)
            # print()
            if adv_label == label:
                score += 1
                scores[t] += 1
            else:
                score -= 1
                scores[t] -= 1
        # print('score :', score)
        if scores[0.1] > 0:
            acc_1 += 1
        if scores[0.2] > 0:
            acc_2 += 1
        if scores[0.3] > 0:
            acc_3 += 1
        if scores[0.1] > 0 and scores[0.2] > 0:
            acc_12 += 1
        if scores[0.1] > 0 and scores[0.3] > 0:
            acc_13 += 1
        if scores[0.2] > 0 and scores[0.3] > 0:
            acc_23 += 1

        if score > 0:
            accuracy += 1

        pbar.set_postfix(accuracy = accuracy/(i+1))

    print("Final Accuracy: ", accuracy / len(og_texts))
    print("Accuracy at 0.1: ", acc_1 / len(og_texts))
    print("Accuracy at 0.2: ", acc_2 / len(og_texts))
    print("Accuracy at 0.3: ", acc_3 / len(og_texts))
    print("Accuracy at 0.1 and 0.2: ", acc_12 / len(og_texts))
    print("Accuracy at 0.1 and 0.3: ", acc_13 / len(og_texts))
    print("Accuracy at 0.2 and 0.3: ", acc_23 / len(og_texts))




    # Original text
    # og_text = "E-mail scam targets police chief Wiltshire Police warns about “phishing” after its fraud squad chief was targeted."
    # text = "E-mail scam targets gendarmerie chief Wiltshire Police warns about “phishing” after its deception battalion massa was targeted."

    # og_text = "aussie ruling party leads in election polls , but gap narrows the government of prime minister john howard had a narrow lead in opinion polls heading into the final week of campaigning ahead of the australian federal election , but the opposition labor party was narrowing the gap , according"
    # text = 	"aussie ruling party leads in election polls , but gap narrows the gov of prime minister john howard had a ltd culminate in opinion polls heading into the final week of campaigning ahead of the aus federal election , but the opposition labor party was downsize the disparity , according"

    # og_text = "I have the good common logical sense to know that oil can not last forever and I am acutely aware of how much of my life in the suburbs revolves around petrochemical products. I’ve been an avid consumer of new technology and I keep running out of space on powerboards - so..."
    # text = "I possess the good common logical sense to realize that oil can not last forever and I am acutely aware of how much of my life in the suburbs spins around petrochemical products. I’ve been an avid consumer of new technology and I keep running out of space on powerboards - well..."
    
    # Add noise with different timesteps
    # timesteps = [0.1, 0.2, 0.3]
    # for t in timesteps:
    #     noised, noisy_tokens = diffusion.add_noise(text, t)
    #     denoised = diffusion.denoise(noisy_tokens, t)
    #     print(f"\nTimestep: {t}")
    #     print(f"Original: {og_text}")
    #     print()
    #     print(f"Adversarial: {text}")
    #     print()
    #     print(f"Noised: {noised}")
    #     print()
    #     print()
        
    #     print(f"Denoised: {denoised}")
    #     print()

if __name__ == "__main__":
    main() 