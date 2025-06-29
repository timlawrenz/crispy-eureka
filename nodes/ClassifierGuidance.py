import torch
import comfy.model_management
import inspect

class ClassifierGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "classifier": ("CLASSIFIER",), # Expects a node that returns a classifier object with a .classify(embedding) method
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_step": ("INT", {"default": 9999, "min": 0, "max": 10000}),
            },
            "optional": {
                "prev_hook": ("LATENT_HOOK",),
            }
        }

    RETURN_TYPES = ("LATENT_HOOK",)
    FUNCTION = "apply_guidance"
    CATEGORY = "latent/guidance"

    def apply_guidance(self, model, clip, vae, classifier, guidance_scale, start_step, end_step, prev_hook=None):
        
        prev_callback = prev_hook[0] if prev_hook is not None else None

        # This hook function contains the core guidance logic.
        # It accepts **kwargs because the hijack passes a variable set of arguments.
        def guidance_hook(**kwargs_in):
            x = kwargs_in['x']

            # --- Chaining Hooks ---
            # If a previous hook exists, call it first.
            if prev_callback is not None:
                # The hijack calls hooks with keyword arguments. We can pass them on.
                x = prev_callback(**kwargs_in)

            step = kwargs_in['step']
            if not (start_step <= step < end_step):
                return x

            # We need these arguments for the guidance calculation.
            sigma = kwargs_in['sigma']
            cond = kwargs_in.get('cond')
            uncond = kwargs_in.get('uncond')
            cond_scale = kwargs_in.get('cond_scale', 1.0) # Default CFG scale

            # --- The Guidance Algorithm ---
            with torch.enable_grad():
                latent_for_grad = x.detach().clone()
                latent_for_grad.requires_grad_()

                # sigma is a tensor, but torch.full expects a scalar
                sigma_tensor = torch.full((latent_for_grad.shape[0],), sigma.item(), device=latent_for_grad.device)

                # Get the noise prediction (epsilon) from the model.
                uncond_pred = model.apply_model(latent_for_grad, sigma_tensor, cond=uncond)
                cond_pred = model.apply_model(latent_for_grad, sigma_tensor, cond=cond)
                noise_pred = uncond_pred + cond_scale * (cond_pred - uncond_pred)
                
                # --- Correctly calculate pred_x0 ---
                # The model predicts noise (epsilon), not the clean image.
                # We derive pred_x0 from the noisy latent (x) and the predicted noise.
                pred_x0 = latent_for_grad - sigma * noise_pred

                # Step 3: Obtain the CLIP Embedding from the *predicted clean image*
                image = vae.decode(pred_x0)
                embedding = clip.encode_image(image)

                # Step 4: Calculate the "Loss" from the classifier
                # The aggregator returns a tensor of scores, with negative classifiers already inverted.
                scores = classifier.classify(embedding.float())
                if scores.numel() == 0: # Check if any classifiers were loaded
                    return x # Skip guidance if no classifiers are active

                # We want to maximize all scores in the tensor.
                # Since negative classifiers already have their scores inverted by the loader,
                # we can use a single, simple loss function.
                loss = -torch.sum(scores)

                # Step 5: Compute the Guidance Gradient
                loss.backward()

            grad = latent_for_grad.grad.detach()
            
            # Normalize the gradient for stable guidance
            grad_norm = torch.linalg.vector_norm(grad, dim=(1, 2, 3), keepdim=True)
            grad = grad / (grad_norm + 1e-7)

            # Step 6: Apply the Gradient to the Denoising Process
            guided_x = x - grad * guidance_scale

            return guided_x

        return ((guidance_hook,),)

NODE_CLASS_MAPPINGS = {
    "ClassifierGuidance": ClassifierGuidance
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClassifierGuidance": "Classifier Guidance"
}
