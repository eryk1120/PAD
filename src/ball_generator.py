import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import pickle
from os import mkdir

from scipy.ndimage import gaussian_filter

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


def normalize_array(array, value_range=(0, 1)):
    """
    Function that takes given array, and normalizes it between given value
    :param array: given array to normalize
    :param value_range: tuple of lower and upper value to normalize array values
    :return: returns normalizes array
    """
    array -= array.min()
    array = array / array.max() * (value_range[1] - value_range[0]) + value_range[0]
    return array


def random_tuple_list(number_of_points=1, border_size=50, im_size=(256, 256), ball_sep_dist=60, ):
    """
    generating list of given length* of 2 element tuplets coresponding to xy positions
    one can specify  distances between generated coordinates, and distance from border

    * generated list can be shorter than given length if minimum can't be fulfield
    :param number_of_points: number of generated points
    :param border_size: minimum distance from egde of the plate
    :param im_size:  size of plane, tuple of xy dimensions eg. (256,256)
    :param ball_sep_dist: minimum distance between points
    :return: gieves list of tuplets coresponding to randomly selected xy coordinates
    """
    coordinates = []
    tries = 0
    for i in range(number_of_points):
        while True:
            x_c, y_c = (random.randint(border_size, im_size[0] - border_size + 1),
                        random.randint(border_size, im_size[1] - border_size + 1))

            distances = [((i[0] - x_c) ** 2 + (i[1] - y_c) ** 2) ** 0.5 for i in
                         coordinates]  # list comprehension for euqiledian distances between freshly generated point
            # and existing ones
            if not distances:
                coordinates.append((x_c, y_c))
                break
            elif min(distances) > ball_sep_dist:
                coordinates.append((x_c, y_c))
                break
            elif tries >= 100:
                tries = 0
                break
            else:
                tries += 1
    return coordinates


def generate_ball():
    """
    generates one plate of balls
    :return:
    """
    # output image properties
    im_size = (256, 256)  # imSize

    # werid ass parameters for phase properties of balls
    sigma_range = (10, 30)  # Bsigm
    same_sigma = False  # same sigma for all of the balls on single image;

    def simga_generator():
        if same_sigma:
            sigma = sigma_range[0] + random.random() * (sigma_range[1] - sigma_range[0])
            while True:
                yield sigma
        while True:
            yield sigma_range[0] + random.random() * (sigma_range[1] - sigma_range[0])

    sigma_generator = simga_generator()

    # choose circle coordinates

    max_no_balls = 10
    no_balls = random.randint(1, max_no_balls)

    data = {"negative": {"coordinates": [], "phase": np.zeros(im_size, dtype=float)},
            "positive": {"coordinates": [], "phase": np.zeros(im_size, dtype=float)}
            }

    no_balls_positive = random.randint(0, no_balls)

    data['negative']['coordinates'] = random_tuple_list(number_of_points=no_balls - no_balls_positive)
    data['positive']['coordinates'] = random_tuple_list(number_of_points=no_balls_positive)

    for coordinate in data['negative']['coordinates']:
        plate = np.zeros(im_size, dtype=float)
        plate[coordinate] = 1
        plate = gaussian_filter(plate, next(sigma_generator))
        plate = -normalize_array(plate)

        data['negative']['phase'] += plate

    for coordinate in data['positive']['coordinates']:
        plate = np.zeros(im_size, dtype=float)
        plate[coordinate] = 1
        plate = gaussian_filter(plate, next(sigma_generator))
        plate = normalize_array(plate)

        data['positive']['phase'] += plate

    data_sample = (data['positive']['phase'] + data['negative']['phase'] + 2,
                   np.stack([data['negative']['phase'], data['positive']['phase']], axis=-1))

    return data_sample


def generate_data(no_images=5, make_pickles=True, save_raw_data=False):
    """
    generates and if chosen,saves data series as pickle or in raw format
    :param no_images: number of images <int>
    :param make_pickles: wether to save series as pickle <bool>
    :param save_raw_data: wether to save raw images<bool>
    :return: NONE
    """

    data = []

    with tqdm(total=no_images, desc='generating data...   ') as pbar:
        with ThreadPoolExecutor(max_workers=cpu_count()) as ex:
            futures = [ex.submit(lambda x: [pbar.update(1), data.append(generate_ball())], pbar) for _ in
                       range(no_images)]
            #                   (^up) lambda for generating image, appending it and updating tqmd progess bar
            for _ in as_completed(futures):
                pass

    def data_sample_to_png(ds, id_time, id_file, outer_pbar):
        input_name = f'../raw_data/{id_time}/merged/{id_file}.png'
        test_name1 = f'../raw_data/{id_time}/testset1/{id_file}.png'
        test_name2 = f'../raw_data/{id_time}/testset2/{id_file}.png'

        plt.imsave(fname=input_name, arr=ds[0])
        plt.imsave(fname=test_name1, arr=ds[1][:, :, 0])
        plt.imsave(fname=test_name2, arr=ds[1][:, :, 1])

        outer_pbar.update(1)

        return 1

    if save_raw_data:
        # saving data to png for veryfication by human
        with tqdm(total=len(data), desc='saving images...     ') as pbar:
            identifier_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

            mkdir(f'../raw_data/{identifier_time}')
            mkdir(f'../raw_data/{identifier_time}/merged')
            mkdir(f'../raw_data/{identifier_time}/testset1')
            mkdir(f'../raw_data/{identifier_time}/testset2')

            with ThreadPoolExecutor(max_workers=cpu_count()) as ex:
                futures = [ex.submit(
                    data_sample_to_png(data[i], id_time=identifier_time, id_file=i,
                                       outer_pbar=pbar)) for i in range(len(data))]
                for _ in as_completed(futures):
                    pass

    if make_pickles:
        file_name = f'../pickles/{datetime.now().strftime("%d.%m.%Y_%H:%M:%S")}.pickle'
        for _ in tqdm(range(int(1)), desc="Making pickles...    "):
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)

    return data


if __name__ == "__main__":
    generate_data(no_images=10, save_raw_data=True, make_pickles=True)
