import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random

from scipy.ndimage import gaussian_filter

from PIL import Image


def generate_data():
    for i in tqdm(range(12), desc='Generating images...'):
        array = np.zeros((512, 512), dtype=np.uint8)
        im = Image.fromarray(array)
        im.save(f"../raw_data/{i}.png")
    print("done")


def normalize_array(array, value_range=(0, 1)):
    array -= array.min()
    array = array / array.max() * (value_range[1] - value_range[0]) + value_range[0]
    return array


def random_tuple_list(number_of_points=1, border_size=50, im_size=(256, 256), ball_sep_dist=60, ):
    coordinates = []
    for i in range(number_of_points):
        while True:
            # TODO that while loop can be infinite, or close to
            #  if given wrong arguments, definitely something to repair, but works for application
            x_c, y_c = (random.randint(border_size, im_size[0] - border_size + 1),
                        random.randint(border_size, im_size[1] - border_size + 1))
            # print(f"x:{x_c}, y:{y_c}")

            distances = [((i[0] - x_c) ** 2 + (i[1] - y_c) ** 2) ** 0.5 for i in
                         coordinates]  # list comprehension for euqiledian distances between freshly generated point
            # and existing ones
            if not distances:
                coordinates.append((x_c, y_c))
                break
            elif min(distances) > ball_sep_dist:
                coordinates.append((x_c, y_c))
                break
    return coordinates


def generate_ball():
    # output image properties
    im_size = (256, 256)  # imSize

    # werid ass parameters for phase properties of balls
    sigma_range = (10, 30)  # Bsigm
    same_sigma = False  # same sigma for all of the balls on single image;

    def simgaGenerator():
        if same_sigma:
            sigma = sigma_range[0] + random.random() * (sigma_range[1] - sigma_range[0])
            while True:
                yield sigma
        while True:
            yield sigma_range[0] + random.random() * (sigma_range[1] - sigma_range[0])

    sigma_generator = simgaGenerator()

    # choose circle coordinates

    max_no_balls = 10
    no_balls = random.randint(1, max_no_balls)

    data = {"negative": {"coordinates": [], "phase": np.zeros(im_size, dtype=float)},
            "positive": {"coordinates": [], "phase": np.zeros(im_size, dtype=float)},
            "merged": {"coordinates": [], "phase": np.zeros(im_size, dtype=float)}
            }

    no_balls_positive = random.randint(0, no_balls)

    data['negative']['coordinates'] = random_tuple_list(number_of_points=no_balls - no_balls_positive)
    data['positive']['coordinates'] = random_tuple_list(number_of_points=no_balls_positive)
    data['merged']['coordinates'] = data['positives']['coordinates'] + data['negative']['coordinates']

    for coordinate in data['negative']['coordinates']:
        sign = random.choice([-1, 1])

        plate = np.zeros(im_size, dtype=float)
        plate[coordinate] = sign
        plate = gaussian_filter(plate, next(sigma_generator))
        plate = normalize_array(plate)

        data['negative']['phase'] -= plate
        data['merged']['phase'] += plate

    for coordinate in data['positive']['coordinates']:
        sign = random.choice([-1, 1])

        plate = np.zeros(im_size, dtype=float)
        plate[coordinate] = sign
        plate = gaussian_filter(plate, next(sigma_generator))
        plate = normalize_array(plate)

        data['positive']['phase'] += plate
        data['merged']['phase'] += plate

    plt.imshow(data['negative']['phase'])
    plt.title("negative")
    plt.colorbar()
    plt.show()

    plt.imshow(data['positive']['phase'])
    plt.title("positive")
    plt.colorbar()
    plt.show()

    plt.imshow(data['merged']['phase'])
    plt.title("merged")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    generate_ball()
