from typing import Literal

from PIL import Image
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class LocalFeatureManager:
    """Manages the local feature space and enable sampling of mixed images.

    Note: there are four spaces of inputs:
    1. feature space: the space of local features (i.e, id of the segments)
    2. ordinal space: the space of integers from (0, 1, ..., num_used_features - 1)
    3. categorical space: the space of matrices, where each row has length num_used_features, and each value is one of choices
    4. one-hot space: the space of one-hot encoded categorical matrices
    """
    def __init__(self, input_image: Image.Image, input_segments: np.ndarray):
        self.input_image = input_image
        self.input_segments = input_segments
        self.im_shape = np.array(input_image).shape
        self.ordinal_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.choices_per_feature = None
        self.binary_masks = None
        self.replacements = None

    def fit(self, 
        choices_per_feature: dict[int, list[int]], 
        replacements: torch.Tensor, 
        binary_masks: torch.Tensor,
    ):
        assert binary_masks.shape[0] == len(replacements), "Number of masks and crops must match"

        self.ordinal_encoder.fit(list(choices_per_feature.keys()))
        self.choices_per_feature = choices_per_feature
        self.binary_masks = binary_masks
        self.replacements = replacements

    def sample_data(self, num_samples: int):
        """Returns two tensors, the one-hot encoded categorical features and the mixed images for regression training."""
        categorical_traits = self.sample_features(num_samples)
        self.categorical_traits = categorical_traits    # TODO: remove
        self.onehot_encoder.fit(categorical_traits)
        onehot_traits = self.onehot_encoder.transform(categorical_traits)
        images = self.mix(categorical_traits)
        return torch.tensor(onehot_traits, dtype=torch.float32), images

    def sample_features(self, num_samples: int):
        """Returns a matrix containing categorical feature vectors."""
        rng = np.random.default_rng(seed=42)
        
        num_features = len(self.choices_per_feature)
        features = np.zeros((num_samples, num_features), dtype=np.int32)
        for f, choices in self.choices_per_feature.items():
            i = self.ordinal_encoder.transform([f])[0]
            features[:, i] = rng.choice(choices, num_samples)

        return torch.tensor(features, dtype=torch.int64)
        
    def mix(self, features: torch.Tensor) -> np.ndarray:
        """Given features (a list of vectors in categorical space), return the mixed images (i.e, the images created by 
        overlaying the replacements according to the binary masks).
        """
        num_replacements = self.binary_masks.shape[0]
        num_samples = features.shape[0]

        indexed_masks = self.binary_masks * torch.arange(num_replacements)[:, None, None]
        blended_masks = torch.take_along_dim(indexed_masks[None, :], features[..., None, None], dim=1).max(dim=1).values
        images = torch.take_along_dim(
            self.replacements.reshape(1, num_replacements, *self.im_shape), 
            blended_masks.reshape(num_samples, 1, *self.im_shape[:2], 1), dim=1,
        ).squeeze(dim=1)
        return images.numpy()

    def decode(self, x: np.ndarray, out_level: Literal["image", "feature", "ordinal", "categorical"], with_choices=False):
        """Given a matrix x, return the decoded tensor at the specified level."""
        assert out_level in ["image", "feature", "ordinal", "categorical"]

        x = self.onehot_encoder.inverse_transform(x)
        if out_level == "categorical": return x
        if out_level == "image": 
            return self.mix(torch.tensor(x, dtype=torch.int64))
        
        base_choices = np.zeros((len(self.choices_per_feature),), dtype=np.int32)
        for i, (_, choices) in enumerate(self.choices_per_feature.items()):
            # we assume the first choice is always the base choice
            base_choices[i] = choices[0]

        sig_features = [np.where(row != base_choices)[0] for row in x]
        sig_with_choices = [
            [(diff_index, row[diff_index]) for diff_index in diff_indices]
            for diff_indices, row in zip(sig_features, x)
        ]

        if out_level == "ordinal": 
            if not with_choices: return sig_features
            return sig_with_choices

        sig_with_choices = [
            [(self.ordinal_encoder.inverse_transform(f), v) for f, v in row]
            for row in sig_with_choices
        ]
        # out_level == "feature"
        if not with_choices: return [[f for f, _ in row] for row in sig_with_choices]
        return sig_with_choices
