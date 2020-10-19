import cv2
from scipy.interpolate import griddata
import numpy as np
from . import radialProfile
from tensorflow import keras


def image_to_features(image_matrix, epsilon=1e-8, number_of_features=722):
  f       = np.fft.fft2(image_matrix)
  fshift  = np.fft.fftshift(f)
  fshift  += epsilon
  magnitude_spectrum  = 20 * np.log(np.abs(fshift))
  psd1D   = radialProfile.azimuthalAverage(magnitude_spectrum)
  points  = np.linspace(0, number_of_features, num=psd1D.size)
  xi      = np.linspace(0, number_of_features, num=number_of_features)
  interpolated  = griddata(points, psd1D, xi, method='cubic')
  interpolated  /= interpolated[0]
  return interpolated

def load_images(image):
  test_images_rgb   = []
  test_images_1chn  = []

  read_img = image.read()

  rgb_image   = cv2.imdecode(np.fromstring(read_img, np.uint8), cv2.IMREAD_UNCHANGED)
  rgb_image   = cv2.resize(rgb_image, (256, 256))
  test_images_rgb.append(rgb_image)

  flt_image   = cv2.imdecode(np.fromstring(read_img, np.uint8), cv2.IMREAD_GRAYSCALE)
  flt_image   = cv2.resize(flt_image, (256, 256))
  interpolated  = image_to_features(flt_image)
  test_images_1chn.append(interpolated)

  return np.asarray(test_images_rgb), np.asarray(test_images_1chn)

def classify_image(img):
  a,b = load_images(img)
  path = "/Users/ahmedalasifer/Desktop/FIT3183.nosync/Server/detect_deepfakes/teamDP"
  model = keras.models.load_model(path)
  result = model.predict([a,b])
  print(result)
  if result < 0.3:
    return 'Fake'
  elif result > 0.6:
    return 'Real'
  else:
    return 'You got me this time!'



