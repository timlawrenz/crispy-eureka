import nodes
import comfy.k_diffusion.sampling
from functools import wraps

# This hijack is based on the method used in https://github.com/kuschanow/ComfyUI-Advanced-Latent-Control

if not hasattr(comfy.k_diffusion.sampling, 'KSAMPLER_ADV_HIJACK'):
    comfy.k_diffusion.sampling.KSAMPLER_ADV_HIJACK = True

    # --- 1. Patch KSamplerAdvanced to accept a hook ---
    k_sampler_adv_class = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvanced']
    original_input_types = k_sampler_adv_class.INPUT_TYPES

    @classmethod
    def hijacked_input_types(cls):
        inputs = original_input_types()
        if 'optional' not in inputs:
            inputs['optional'] = {}
        inputs['optional']['latent_hook'] = ('LATENT_HOOK',)
        return inputs

    k_sampler_adv_class.INPUT_TYPES = hijacked_input_types

    # --- 2. Patch KSamplerAdvanced.sample to manage the hook ---
    original_KSamplerAdvanced_sample = k_sampler_adv_class.sample

    @wraps(original_KSamplerAdvanced_sample)
    def hijacked_KSamplerAdvanced_sample(self, model, *args, latent_hook=None, **kwargs):
        if latent_hook is None:
            return original_KSamplerAdvanced_sample(self, model, *args, **kwargs)

        # Attach hook to the model object for later retrieval in the sampling function
        setattr(model, 'latent_hook', latent_hook[0])
        try:
            return original_KSamplerAdvanced_sample(self, model, *args, **kwargs)
        finally:
            if hasattr(model, 'latent_hook'):
                delattr(model, 'latent_hook')

    k_sampler_adv_class.sample = hijacked_KSamplerAdvanced_sample

    # --- 3. Patch k-diffusion samplers to apply the hook via a callback ---
    def create_sampler_wrapper(original_sampler_func):
        @wraps(original_sampler_func)
        def wrapper(model, x, sigmas, *args, **kwargs):
            hook_func = getattr(model.inner_model, 'latent_hook', None)
            if hook_func is None:
                return original_sampler_func(model, x, sigmas, *args, **kwargs)

            import inspect
            sig = inspect.signature(hook_func)
            
            original_callback = kwargs.get('callback', None)
            
            def new_callback(data):
                # Prepare arguments for the hook function based on its signature.
                # This allows for simple hooks (e.g., def hook(x)) and advanced hooks.
                hook_kwargs = {'x': data['x']}
                
                # Check if the hook accepts **kwargs to pass everything
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    hook_kwargs.update(kwargs.get('extra_args', {}))
                    hook_kwargs['step'] = data['i']
                    hook_kwargs['sigma'] = data['sigma']
                else: # Otherwise, pass only the parameters it explicitly asks for
                    if 'step' in sig.parameters:
                        hook_kwargs['step'] = data['i']
                    if 'sigma' in sig.parameters:
                        hook_kwargs['sigma'] = data['sigma']
                    if any(p in sig.parameters for p in ['cond', 'uncond', 'cond_scale']):
                        extra_args = kwargs.get('extra_args', {})
                        if 'cond' in sig.parameters: hook_kwargs['cond'] = extra_args.get('cond')
                        if 'uncond' in sig.parameters: hook_kwargs['uncond'] = extra_args.get('uncond')
                        if 'cond_scale' in sig.parameters: hook_kwargs['cond_scale'] = extra_args.get('cond_scale')

                # Call the hook and update the latent
                data['x'] = hook_func(**hook_kwargs)
                
                if original_callback is not None:
                    original_callback(data)
            
            kwargs['callback'] = new_callback
            return original_sampler_func(model, x, sigmas, *args, **kwargs)
        return wrapper

    # Patch all known k-diffusion samplers
    samplers_to_patch = [
        'sample_euler', 'sample_euler_ancestral', 'sample_heun', 'sample_heunpp2', 
        'sample_dpm_2', 'sample_dpm_2_ancestral', 'sample_lms', 'sample_dpm_fast', 
        'sample_dpm_adaptive', 'sample_dpmpp_2s_ancestral', 'sample_dpmpp_sde', 
        'sample_dpmpp_2m', 'sample_dpmpp_2m_sde', 'sample_ddim', 'sample_uni_pc', 'sample_uni_pc_bh2'
    ]
    for name in samplers_to_patch:
        if hasattr(comfy.k_diffusion.sampling, name):
            setattr(comfy.k_diffusion.sampling, name, create_sampler_wrapper(getattr(comfy.k_diffusion.sampling, name)))

    print("Successfully hijacked KSamplerAdvanced for latent control.")
