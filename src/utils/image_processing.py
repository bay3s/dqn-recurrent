import numpy as np


def image_rgb_to_gray(rgb: np.ndarray):
  """
  Compute luminance of an RGB image.

  Parameters
  ----------
  rgb : (..., 3, ...) array_like
      The image in RGB format. By default, the final dimension denotes
      channels.

  Returns
  -------
  out : ndarray
      The luminance image - an array which is the same size as the input
      array, but with the channel dimension removed.

  Notes
  -----
  The weights used in this conversion are calibrated for contemporary
  CRT phosphors::
      Y = 0.2125 R + 0.7154 G + 0.0721 B
  If there is an alpha channel present, it is ignored.

  Examples
  --------
    > img_gray = rgb2gray(img)
  """
  coeffs = np.array([0.2125, 0.7154, 0.0721], dtype = rgb.dtype)

  return rgb @ coeffs
