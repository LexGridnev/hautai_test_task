from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


class PixelClustering:
    """Class for pixel clustering algorithm. Uses K-Means algorithm for
    pixel clustering."""
    def __init__(
            self,
            channel_indices: Optional[Tuple[int, ...]] = None,
            coordinate_system: Optional[str] = None
    ) -> None:
        """
        :param channel_indices: Indices of image channels to use for
        clustering. For example you have RGB image and want to use only R and
        B channels for clustering. In this case you should pass (0, 2) into
        this parameter;
        :param coordinate_system: The name of the coordinate system in which the
        pixel coordinates will be represented. This is necessary if we want to
        use pixel coordinates as additional features. Should be one of
        ('euclidean', 'polar').
        """
        self._clustering = KMeans(n_clusters=8)
        self._channel_indices = channel_indices
        self._coordinate_system = coordinate_system

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image by constructing vectors from pixels. The constructed
        vectors will have the same number of components as the channels in the
        original image.

        :param image: Numpy array of shape HxWxC (Height x Width x Channels).

        :return: Numpy array of shape H*W x C - vectors from image pixels.
        """
        if image.ndim < 2:
            raise ValueError(
                'Algorithm available only for image (Array with ndim >= 2).'
            )
        preprocessed = image.copy()

        # Pick specific channels for clustering.
        if self._channel_indices:
            channels = preprocessed[:, :, self._channel_indices[0]]
            for channel_index in self._channel_indices[1:]:
                try:
                    channels = np.dstack(
                        (channels, preprocessed[:, :, channel_index])
                    )
                except IndexError:
                    # Pick only those channels that the original image has.
                    continue
            preprocessed = channels

        # Ensure the image has at least one channel.
        if preprocessed.ndim == 2:
            preprocessed = np.expand_dims(preprocessed, axis=2)
        image_height, image_width, image_channels = preprocessed.shape
        image_vectors = preprocessed.reshape(-1, image_channels)

        if not self._coordinate_system:
            return image_vectors
        else:
            # Pixel's coordinates in euclidean system centered at image center.
            coordinates = np.array(
                [
                    (
                        np.tile(np.arange(0, image_width), image_height)
                        - image_width // 2
                    ),
                    (
                        np.repeat(np.arange(0, image_height), image_width)
                        - image_height // 2
                    )
                ]
            ).T
        if self._coordinate_system == 'euclidean':
            image_vectors = np.hstack((image_vectors, coordinates))
        elif self._coordinate_system == 'polar':
            phi = np.arctan2(
                coordinates[:, 1], coordinates[:, 0]
            ).reshape(-1, 1)
            image_vectors = np.hstack((image_vectors, phi))
        else:
            raise ValueError(
                'Coordinate system should be one of (\'euclidean\', \'polar\')'
            )
        return image_vectors

    def fit(self, image: np.ndarray) -> None:
        """
        Fit clustering algorithm on given image.

        :param image: Numpy array of shape HxWxC (Height x Width x Channels).
        """
        self._clustering.fit(self._preprocess_image(image))

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict labels for pixels of given image.

        :param image: Numpy array of shape HxWxC (Height x Width x Channels).

        :return: Numpy array of shape HxW - image mask with per pixel labels.
        """
        image_height, image_width, _ = image.shape
        return self._clustering.predict(
            self._preprocess_image(image)
        ).reshape(image_height, image_width)

    def fit_predict(self, image: np.ndarray) -> np.ndarray:
        """
        Fit clustering algorithm on given image and predict labels for it's
        pixels.

        :param image: Numpy array of shape HxWxC (Height x Width x Channels).

        :return: Numpy array of shape HxW - image mask with per pixel labels.
        """
        self.fit(image)
        return self.predict(image)
