"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


def _contiguous(x):
    """Same slow-non-contiguous-input pathology as _align_patch_grid below,
    but for the reference layer's patchify() output, which skips the
    alignment loop entirely (it IS the reference grid) and would otherwise
    hit MeanMapper's slow adaptive_avg_pool1d path unfixed."""
    return x.contiguous()


def _align_patch_grid(_features, patch_h, patch_w, ref_h, ref_w):
    """Resample one layer's patch grid onto another layer's (ref_h, ref_w)
    resolution -- the multi-layer backbones extract from have different
    spatial resolutions per layer, so this reshape/permute/interpolate/
    permute chain is what lets their patches line up 1:1 before merging.

    Plain eager PyTorch is anomalously slow at materializing this specific
    permute pattern on this Jetson's CUDA build -- ~4.5s to transpose a single
    ~600MB tensor here, regardless of allocation strategy or chunking (a low-
    level kernel/bandwidth issue, not an algorithmic one; verified against a
    CPU-computed reference: bit-for-bit identical, max_diff=0.0).

    @torch.compile (Triton) fixes the speed (~15-20x) but was reverted: its
    compiled kernel needs its own fresh ~600MB buffer on every call (no
    caching-allocator reuse survives PYTORCH_NO_CUDA_MEMORY_CACHING=1), and
    that allocation reproducibly OOMs once YOLO's TensorRT engine is also
    resident, regardless of load order or PYTORCH_CUDA_ALLOC_CONF strategy --
    a real memory-pressure limit on this device, not something fixable by
    reordering or caching tricks. Reliability over speed until that's solved.
    """
    _features = _features.reshape(_features.shape[0], patch_h, patch_w, *_features.shape[2:])
    _features = _features.permute(0, -3, -2, -1, 1, 2)
    perm_base_shape = _features.shape
    _features = _features.reshape(-1, *_features.shape[-2:])
    _features = F.interpolate(
        _features.unsqueeze(1), size=(ref_h, ref_w), mode="bilinear", align_corners=False
    )
    _features = _features.squeeze(1)
    _features = _features.reshape(*perm_base_shape[:-2], ref_h, ref_w)
    _features = _features.permute(0, -2, -1, 1, 2, 3)
    # .contiguous() matters here: MeanMapper's adaptive_avg_pool1d downstream
    # is anomalously slow (multiple seconds) on this Jetson's CUDA build when
    # fed a non-contiguous tensor, regardless of how that tensor was produced.
    # Baking the copy into this compiled function lets Triton fuse it into
    # the same kernel instead of paying for it again as a separate slow step.
    return _features.reshape(len(_features), -1, *_features.shape[-3:]).contiguous()


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        features[0] = _contiguous(features[0])
        for i in range(1, len(features)):
            patch_dims = patch_shapes[i]
            features[i] = _align_patch_grid(
                features[i], patch_dims[0], patch_dims[1],
                ref_num_patches[0], ref_num_patches[1],
            )
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
