"""
Functions for explaining classifiers that use Image data.
"""
import logging
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

from lime.lime_base import LimeBase
from lime.lime_image import LimeImageExplainer, ImageExplanation
from lime.wrappers.scikit_image import SegmentationAlgorithm

from exrep.utils import torch_pairwise_cosine_similarity

logger = logging.getLogger(__name__)

class RepresentationLimeImageExplainer(LimeImageExplainer):
    def generate_local_data(self, image, classifier_fn, 
                            hide_color=None,
                            num_samples=1000,
                            batch_size=10,
                            segmentation_fn=None,
                            random_seed=None,
                            progress_bar=None):
        """Generates data for local model fitting.

            Args:
                classifier_fn: classifier prediction probability function, which
                    takes a numpy array and outputs prediction probabilities.  For
                    ScikitClassifiers , this is classifier.predict_proba.
                hide_color: If not None, will hide superpixels with this color.
                    Otherwise, use the mean pixel color of the image.
                segmentation_fn: SegmentationAlgorithm, wrapped skimage
                    segmentation function
                random_seed: integer used as random seed for the segmentation
                    algorithm. If None, a random integer, between 0 and 1000,
                    will be generated using the internal random number generator.
                progress_bar: if True, show tqdm progress bar.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        segments = segmentation_fn(image)

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

    
        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        self.cache = {'data': data, 'labels': labels, 'segments': segments}
        return data, labels, segments

    def explain_instance(self, image, directions, 
                         num_features=1000,
                         labels=(1,),
                         distance_metric='cosine',
                         temperature=1.0,
                         model_regressor=None,):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """
        top = labels
        
        if not hasattr(self, 'cache'):
            raise ValueError("Must generate local data before explaining instance")
        else:
            logger.info("Using cached perturbation data")
            data = self.cache['data']
            labels = self.cache['labels']
            segments = self.cache['segments']

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        
        print(f"Labels = {top}")
        # (ret_exp.intercept[label],
        #     ret_exp.local_exp[label],
        #     ret_exp.score[label],
        #     ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
        #     data, labels, distances, label, num_features,
        #     model_regressor=model_regressor,
        #     feature_selection=self.feature_selection)
        
        (_, _, _, _) = self.base.explain_instance_with_data(
            data, labels, distances, top, num_features,
            model_regressor=model_regressor,
            feature_selection=self.feature_selection)
        
        # intercepts have shape (n_features, )
        # coef has shape (n_features, n_patches)
        # image embedding has shape (n_features, )
        # directions are (n_labels, n_features)
        n_local_features = data.shape[1]
        projected_dirs = model_regressor.project(directions)
        
        # need to compute derivative of prob(image_embedding | projected_dirs.T) w.r.t. the patches
        import torch
        device = "cuda"
        ones = torch.ones((1, n_local_features), requires_grad=True, device=device, dtype=torch.float32)
        x = model_regressor.embed(ones)
        y = torch.tensor(projected_dirs, device=device, dtype=torch.float32)
        sim = torch_pairwise_cosine_similarity(x, y)[0]
        prob = torch.nn.functional.softmax(sim / temperature, dim=0)
        print("Sim shape = ", sim.shape)

        used_features = list(range(n_local_features))
        for label, _ in enumerate(directions):
            attrib_values = (torch.autograd.grad(prob[label], ones, retain_graph=True)[0]).detach().cpu().numpy()[0]
            print("Attrib shape = ", attrib_values.shape)
            ret_exp.intercept[label] = None
            ret_exp.local_exp[label] = sorted(zip(used_features, attrib_values), key=lambda x: np.abs(x[1]), reverse=True)
            ret_exp.score[label] = None
            ret_exp.local_pred[label] = None
        
        return ret_exp


        # directions = exp_dirs
        # image_embedding = model_regressor.predict(np.ones((1, data.shape[1])))
        # # need to compute derivative of image_embedding @ directions.T w.r.t. the patches
        # # logits have shape (n_labels, 1)
        # approx_logit = np.linalg.norm(image_embedding - directions, axis=1, keepdims=True)
        # # attrib_values has shape (n_patches, n_labels)
        # attrib_values = model_regressor.coef_.T @ ((image_embedding - directions) / approx_logit).T

# class KNearestNeighborEmbedding:
#     def __init__(self, n_neighbors=10, kernel_fn=None, temperature=1.0, random_state=None):
#         self.kernel_fn = kernel_fn
#         self.temperature = temperature
#         self.random_state = check_random_state(random_state)
#         self.knn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors)

#     def fit(self, x, y):
#         self.knn.fit(x)
#         self.targets = y

#     def predict(self, x):
#         dist, ind = self.knn.kneighbors(x)

        
#         # the interpretable directions live in the h \circ e space
#         # we will find its nearest neighbors in that space, then
#         # find the corresponding images, and compute the image of the 
#         # directions in the approximated space as the weighted average of 
#         # the k-nearest neighbors
        
#         neigh_dist, neigh_ind = knn.kneighbors(directions)
#         # convert distances to similarities
#         print("Neighbor distances = ", neigh_dist)
#         neigh_sim = np.exp(-neigh_dist)
#         neigh_sim /= np.sum(neigh_sim, axis=1, keepdims=True)
#         print("Neighbor similarities = ", neigh_sim)
#         # the following has shape (n_labels, n_features)
#         exp_dirs = np.sum(labels[neigh_ind] * neigh_sim[:, :, np.newaxis], axis=1)