import torch
import torch.nn as nn
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch.nn.init as init

import torch.optim as optim

class PrefixToken(nn.Module):
    def __init__(self, share_token_num):
        super().__init__()

        self.projection = nn.Linear(768, 768)
        self.share_token = nn.Parameter(torch.randn(share_token_num, 768))

        self.projection.requires_grad_(True)
        self.share_token.requires_grad_(True)

        # Initialize the weights of the linear layer
        self._initialize_weights()

    def _initialize_weights(self):
        # Using Xavier initialization (Glorot) for the weights
        init.xavier_uniform_(self.projection.weight)

        # Initialize the bias with zeros
        if self.projection.bias is not None:
            init.zeros_(self.projection.bias)

        # Initialize the shared token parameter with a normal distribution
        init.normal_(self.share_token, mean=0.0, std=0.02)

    def forward(self, image, clip_model, clip_processor,text_embeddings=None,sampling_strategy='token_and_text', sub=1):
        if sampling_strategy == 'text':
            hidden = text_embeddings.to(self.projection.weight.device)
            unshare_token = self.projection(hidden)
        elif sampling_strategy == 'token_and_text':
            inputs = clip_processor(images=image, return_tensors="pt")
            vision_outputs = clip_model.vision_model(
                pixel_values=inputs['pixel_values'].to('cuda'),
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = vision_outputs['last_hidden_state']  # B, 50, dim
            if sub > 1:
                hidden = torch.cat([hidden[:, :1, :], hidden[:, 1::sub, :]], dim=1)
            hidden = hidden.to(self.projection.weight.device)
            unshare_token = self.projection(hidden)

        prefix_token = torch.cat([self.share_token.unsqueeze(0).expand(hidden.size(0), -1, -1), unshare_token], dim=1)

        return prefix_token

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path, device):
        self.load_state_dict(torch.load(file_path, map_location=device))


def test():
    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load an image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Initialize the PrefixToken module

    # Ensure the model is on the same device as the image tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)

    prompt_learner = PrefixToken(4).cuda()

    # Forward pass through the PrefixToken module
    output_prefix_token = prompt_learner(image, clip_model, clip_processor)
    print(output_prefix_token.shape)

    # Save the module
    save_path = "prefix_token_model.pth"
    prompt_learner.save(save_path)
    print(f"Model saved to {save_path}")

    # Load the module
    loaded_prompt_learner = PrefixToken(4)
    loaded_prompt_learner.load(save_path, device)
    loaded_prompt_learner.to(device)
    print("Model loaded successfully")

    # Verify that the loaded model gives the same output
    loaded_output_prefix_token = loaded_prompt_learner(image, clip_model, clip_processor)
    print(torch.allclose(output_prefix_token, loaded_output_prefix_token, atol=1e-6))

    print(loaded_output_prefix_token.shape)

    loaded_output_prefix_token = loaded_prompt_learner(image, clip_model, clip_processor, sub=2) # subsample
    print(loaded_output_prefix_token.shape)

    # Define an optimizer, including the parameters of PromptLearner
    optimizer = optim.Adam(loaded_prompt_learner.parameters(), lr=1e-3)

    # Example usage in a training loop
    # Fake data for demonstration

    texts = ["This is a sample text for testing purposes"]  # 示例文本

    losses = []
    for epoch in range(100):
        # Zero gradients
        optimizer.zero_grad()

        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        num_tokens = inputs['input_ids'].shape[1]
        if num_tokens > 10:
            sampling_strategy = 'token_and_text'
        else:
            sampling_strategy = 'text'
        data = torch.randn(1, 224, 224, 3).clip(0, 1).to(device)

        prefix_tokens = loaded_prompt_learner(data, clip_model, clip_processor,inputs['input_ids'],sampling_strategy, sub=2) # subsample

        ## sd inference, concat the prefix token with the text embedding, note that the maximum tokens is set to 77, you might need to set larger value in clip config
        # final_text_embedding = torch.cat([prefix_tokens, text_embdding], dim=1)
        # stable_diffusion(xxxx, final_text_embedding)
        ##

        ## the following is loss is not for usage, just for backward example

        loss = (prefix_tokens).mean()

        losses.append(loss.cpu().item())

        loss.backward()

        # Print gradients of each parameter
        print(f"Epoch {epoch+1} gradients:")
        for name, param in loaded_prompt_learner.named_parameters():
            if param.grad is not None:
                print(f"{name}.grad: {param.grad}")

        optimizer.step()

    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig('loss.png')
    print(losses)


if __name__ == '__main__':
    test()

