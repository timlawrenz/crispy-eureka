import torch
import pickle
import os
import comfy.utils

# --- Configuration ---
# Place your .pkl classifier files in this directory
CLASSIFIER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "models", "classifiers")

# This is the class that will be returned by the Loader node.
# It holds multiple classifiers and knows whether they are positive or negative.
class ClassifierAggregator:
    def __init__(self, positive_paths, negative_paths):
        self.positive_classifiers = self.load_classifiers(positive_paths)
        self.negative_classifiers = self.load_classifiers(negative_paths)

    def load_classifiers(self, paths):
        """Loads a list of .pkl files from the specified directory."""
        loaded_classifiers = []
        if not paths:
            return loaded_classifiers
            
        for path in paths:
            full_path = os.path.join(CLASSIFIER_DIR, path.strip())
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'rb') as f:
                        # Use a progress bar for loading large models
                        pbar = comfy.utils.ProgressBar(len(f.read()))
                        f.seek(0)
                        clf = pickle.load(f)
                        loaded_classifiers.append(clf)
                        print(f"Successfully loaded classifier: {path}")
                except Exception as e:
                    print(f"Error loading classifier {path}: {e}")
            else:
                print(f"Classifier not found: {full_path}")
        return loaded_classifiers

    def classify(self, embedding):
        """
        This method will be called by the ClassifierGuidance node.
        It runs all loaded classifiers and returns a combined score tensor.
        """
        device = embedding.device
        batch_size = embedding.shape[0]
        all_scores = []

        # Ensure embedding is a numpy array on the CPU for scikit-learn
        embedding_np = embedding.cpu().numpy()

        # Get scores for positive classifiers
        for clf in self.positive_classifiers:
            # Assumes scikit-learn classifier with predict_proba
            # We take the probability of the "True" class (index 1)
            score = clf.predict_proba(embedding_np)[:, 1]
            all_scores.append(torch.tensor(score, device=device, dtype=torch.float32))

        # Get scores for negative classifiers and negate them
        for clf in self.negative_classifiers:
            score = clf.predict_proba(embedding_np)[:, 1]
            # By negating the score here, we make the guidance node's job easier.
            # Maximizing a negative score is the same as minimizing the original score.
            all_scores.append(torch.tensor(-score, device=device, dtype=torch.float32))

        if not all_scores:
            # Return a zero tensor if no classifiers were loaded
            return torch.zeros((batch_size, 0), device=device)

        # Stack all scores into a single tensor
        return torch.stack(all_scores, dim=-1)


class ClassifierLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Ensure the classifier directory exists
        os.makedirs(CLASSIFIER_DIR, exist_ok=True)
        
        return {
            "required": {
                "positive_classifiers": ("STRING", {"multiline": True, "default": "# One classifier filename per line\n"}),
                "negative_classifiers": ("STRING", {"multiline": True, "default": "# e.g., black_and_white.pkl\n"}),
            }
        }

    RETURN_TYPES = ("CLASSIFIER",)
    FUNCTION = "load_and_aggregate"
    CATEGORY = "latent/guidance"

    def load_and_aggregate(self, positive_classifiers, negative_classifiers):
        # Split multiline string into a list of filenames
        positive_paths = [name for name in positive_classifiers.splitlines() if name.strip() and not name.strip().startswith('#')]
        negative_paths = [name for name in negative_classifiers.splitlines() if name.strip() and not name.strip().startswith('#')]

        aggregator = ClassifierAggregator(positive_paths, negative_paths)
        return (aggregator,)

# Add the new node to ComfyUI's mappings
NODE_CLASS_MAPPINGS = {
    "ClassifierLoader": ClassifierLoader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClassifierLoader": "Classifier Loader"
}
