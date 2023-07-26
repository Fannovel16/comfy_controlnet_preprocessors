import torch
from random import randint
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.sample import prepare_noise


def preprocess(style_fidelity, ref, model):
    def attn1_patch(q, k, v, extra_options):
        nonlocal is_self_attention
        i = extra_options["transformer_index"]
        attn_weight = float(i)/float(num_attn_modules)
        # Check for self-attention
        if torch.equal(q, k) and torch.equal(k, v):
            is_self_attention = True

            if MODE == "write":
                bank.append(q.detach().clone())

            if MODE == "read":
                if style_fidelity > attn_weight and len(bank) > 0:
                    kv = torch.cat((q, bank[i]), dim=1)
                    return q, kv, kv
                
        return q, k, v
    
    def attn1_output_patch(q, extra_options):
        nonlocal is_self_attention
        attn_weight = float(extra_options["transformer_index"])/float(num_attn_modules)

        if is_self_attention and MODE == "read" and style_fidelity > attn_weight:
            q = style_fidelity * q + (1.0 - style_fidelity) * q
            is_self_attention = False

        return q

    def unet_function_wrapper(model_function, params):
        def get_noise_level():
            """
            returns a scalar value between 0 and 1 that represents the noise level based on timesteps.
            """
            # one possible formula is linear decay: noise_level = 1 - timesteps / max_timesteps
            max_timesteps = 999 # assuming this is the maximum value of timesteps
            noise_level = 1 - torch.mean(timestep) / max_timesteps # take the mean of timesteps tensor and apply linear decay formula
            return noise_level.to(torch.device("cpu"))
        
        nonlocal MODE, ref
        timestep = params["timestep"].to(model.model.diffusion_model.dtype)
        input_x = params["input"]
        c = params["c"]

        # get the noise level from timesteps
        noise_level = get_noise_level()

        if not torch.all(timestep == 0):  # check if it is not the last call
            noise = noise_level * original_noise
            ref_x = noise + ref  # noising up image.
        else:
            ref_x = ref

        # ref_x = ref_x.to(input_x.device)
        # ref = ref.to(input_x.device)
        ref_x = torch.cat((ref_x, ref))

        MODE = "write"

        model_function(ref_x, timestep, **c)

        MODE = "read"

        output = model_function(input_x, timestep, **c)
        bank.clear()

        return output

    def torch_dfs(model):
        result = [model]
        for child in model.children():
            result += torch_dfs(child)
        return result
    
    if style_fidelity == 0:
        return (model,)

    ref = model.model.process_latent_in(ref["samples"])
    original_noise = prepare_noise(ref, randint(0, 18446744073709552000))
    num_attn_modules = len([module for module in torch_dfs(model.model) if isinstance(module, BasicTransformerBlock)])
    bank = []
    is_self_attention = False

    MODE = "write"

    reference_only_model = model.clone()
    reference_only_model.set_model_unet_function_wrapper(unet_function_wrapper)
    reference_only_model.set_model_attn1_patch(attn1_patch)
    reference_only_model.set_model_attn1_output_patch(attn1_output_patch)

    return (reference_only_model,)
