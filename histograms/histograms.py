import matplotlib
import matplotlib.pyplot as plt

import random
from util.file_storage import *

def get_hist(images, bin_count = 64):
  n,h,w,c = images.shape
  concat_image = images.reshape((n*h, w, c))
  
  b_hist = cv2.calcHist([concat_image], [0], None, [bin_count], [0, 256], accumulate=False)
  g_hist = cv2.calcHist([concat_image], [1], None, [bin_count], [0, 256], accumulate=False)
  r_hist = cv2.calcHist([concat_image], [2], None, [bin_count], [0, 256], accumulate=False)
    
  total_pixels = n*h*w
  r_hist /= total_pixels
  g_hist /= total_pixels
  b_hist /= total_pixels
  
  fig = plt.figure()
  plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
  x = np.arange(0, 256, (256 / bin_count))
  
  plt.plot(x, b_hist, color='#0000FF')
  plt.plot(x, g_hist, color='#00FF00')
  plt.plot(x, r_hist, color='#FF0000')
  plt.minorticks_on()

  plt.xlabel("Brightness")
  plt.ylabel("Frequency")
  if len(images) == 1:
    plt.title("RGB Histogram")
  else:
    plt.title(f"RGB Histogram ({len(images)} Images)")
  fig.canvas.draw()

  # return as cv image
  data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
  w, h = fig.canvas.get_width_height()
  return cv2.cvtColor(data.reshape([h, w, 4]), cv2.COLOR_RGBA2BGR)

def factor_image(image, brightness_factor):
  return image.add_(1.0).mul_(brightness_factor).sub_(1.0)

def add_noise(image, std_dev):
  noise = torch.normal(image, torch.full(image.shape, std_dev))
  return torch.clamp(noise, -1, 1)

def export_all_histogram(image_names):
  images_count = len(image_names)

  images_low = np.empty((images_count, 400, 600, 3), dtype=np.uint8)
  images_normal = np.empty((images_count, 400, 600, 3), dtype=np.uint8)

  for i, name in enumerate(image_names[:images_count]):
    low, normal = dataset.get_image_pair(object_storage, name)
    images_low[i] = low
    images_normal[i] = normal
    print(f"{i+1}/{len(image_names)}")

  hist_low = get_hist(images_low, bin_count = 128)
  hist_normal = get_hist(images_normal, bin_count = 128)

  cv2.imwrite("histograms/hist_low.png", hist_low)
  cv2.imwrite("histograms/hist_normal.png", hist_normal)

  cv2.waitKey()

userdata = JsonUserdata("userdata.json")
object_storage = S3ObjectStorage(userdata)
dataset = Dataset("datasets/lowlight/LOL-v2/Real_captured/Train/Low/low",
                  "datasets/lowlight/LOL-v2/Real_captured/Train/Normal/normal")

image_names = dataset.get_names(object_storage)
random.shuffle(image_names)

low, normal = dataset.get_image_pair(object_storage, image_names[0])

cv2.imshow("Normal", normal)
cv2.imshow("Low", low)

mean_normal = normal.mean()
mean_low = low.mean()

tensor = image_to_tensor(low)

tensor = factor_image(tensor, mean_normal / mean_low)

image_out = tensor_to_images(tensor)[0]

cv2.imshow("Enhanced", image_out)

cv2.waitKey()