from typing import Iterable, Optional, TypeVar, Union

from torch import Tensor

from lsat.typing.data_formats import KeypointData


CLIP_HINT = TypeVar('CLIP_HINT')
KEYPOINTS_HINT = TypeVar('KEYPOINTS_HINT')
LABEL_HINT = TypeVar('LABEL_HINT')

Sample = tuple[
    Optional[Union[Iterable[Tensor], CLIP_HINT]],
    Optional[Union[Iterable[KeypointData], KEYPOINTS_HINT]],
    Union[str, LABEL_HINT]
]
