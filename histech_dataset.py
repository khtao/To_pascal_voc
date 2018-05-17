import os
from glob import glob
import numpy as np
import scipy.io as scio
from tifffile import TiffFile


def ReadTiff(filepath):
    with TiffFile(filepath) as tif:
        images = tif.asarray().astype(np.uint8)
    return images


def load_target(filepath):
    """
    :param filepath images path:
    :return bbox label:
    'ASCUS' class 0
    'LSIL' class 1
    'HSIL class 2
    """
    # filepath = filepath.replace("im_scale", "gt_scale")
    data = scio.loadmat(filepath[:-4] + ".mat")
    data = data["bbox"].astype(np.int32)
    bbox = data[:, 1:]
    label = data[:, 0].astype(np.int32)
    label = np.reshape(label, len(label))
    return bbox, label


def make_dataset(dir, mode):
    images_path = os.path.join(dir, mode)
    image_list = glob(os.path.join(images_path, "*.tif"))
    return image_list


class HistechDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"

    """

    def __init__(self, root='.', mode="train", transform=None, ):
        self.ids = make_dataset(root, mode)
        self.data_dir = root
        self.label_names = Histech_BBOX_LABEL_NAMES
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        bbox, label = load_target(id_)
        bbox = bbox + 1
        bbox = np.maximum(np.minimum(bbox, 640), 1)
        img = ReadTiff(id_)
        if self.transform is not None:
            img, bbox = self.transform(img, bbox)
        # temp = np.copy(bbox)
        # bbox[:, 0] = temp[:, 1]
        # bbox[:, 1] = temp[:, 0]
        # bbox[:, 2] = temp[:, 3]
        # bbox[:, 3] = temp[:, 2]
        return img, bbox, label

    __getitem__ = get_example


Histech_BBOX_LABEL_NAMES = (
    'ascus',
    'lsil',
    'hsil',)
