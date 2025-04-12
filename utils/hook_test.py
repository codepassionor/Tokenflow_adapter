from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

class DistFeatureExtractorV2:
    def __init__(self, model, accelerator):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()
        self.accelerator = accelerator

    def hook_function(self, module, input, output):
        output_gathered = self.accelerator.gather(output)
        if self.accelerator.is_main_process:
            #print('output_gathered.shape')
            #print(output_gathered.shape)
            self.features.append(output_gathered)

    def register_hook(self):
        #print(self.model._modules)
        res_dict = {1: [1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        unet = self.model
        hooks = []
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self):
        #_ = self.model(input_data)
        return self.features

    def reset(self):
        del self.features
        self.features = []

    def remove_hook(self):
        self.hook.remove()

class DistFeatureExtractor:
    def __init__(self, model, accelerator):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()
        self.accelerator = accelerator

    def hook_function(self, module, input, output):
        output_gathered = self.accelerator.gather(output)
        if self.accelerator.is_main_process:
            #print('output_gathered.shape')
            #print(output_gathered.shape)
            self.features.append(output_gathered)

    def register_hook(self):
        #print(self.model._modules)
        res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        unet = self.model
        hooks = []
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self):
        #_ = self.model(input_data)
        return self.features

    def reset(self):
        self.features = []

    def remove_hook(self):
        self.hook.remove()


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hooks = []
        self.register_hook()

    def hook_function(self, module, input, output):
        self.features.append(output)

    def register_hook(self):
        #print(self.model._modules)
        res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
        unet = self.model
        hooks = []
        
        for res in res_dict:
            for block in res_dict[res]:
                module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
                self.hooks.append(module.register_forward_hook(self.hook_function))

    def get_features(self):
        #_ = self.model(input_data)
        return self.features

    def reset(self):
        del self.features
        self.features = []

    def remove_hook(self):
        self.hook.remove()

if __name__ == '__main__':
    load_model = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9', subfolder="unet").cuda()
    hooker = FeatureExtractor(load_model)

    import torch
    data = torch.zeros((1,4, 64, 64)).cuda()
    t = torch.zeros([1]).long().cuda()
    load_model(data, t, encoder_hidden_states=torch.zeros(1,77, 768).cuda())
    from IPython import embed; embed()

