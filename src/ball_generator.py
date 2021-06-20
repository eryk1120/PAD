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

def generate_ball():
    # output image properties
    im_size = (256, 256)  # imSize

    # balls geometrical placement
    border_size = 50 # distance between balls and edge of image
    ball_sep_dist = 60 # minimal distance between adjacent balls in positive/negative mask

    # werid ass parameters for phase properties of balls
    sigma_range = (10, 30)  # Bsigm
    same_sigma = False # same sigma for all of the balls on single image;

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
    coordinates =[]

    # picking coordinates for balls
    for i in range( random.randint(1,max_no_balls)):
        while True:
            #TODO that while loop can be infinite, or close to if given wrong arguments, definitely something to repair, but works for application
            x_c, y_c = (random.randint(border_size, im_size[0]-border_size+1),
                                  random.randint(border_size, im_size[1]-border_size+1))
            #print(f"x:{x_c}, y:{y_c}")

            distances = [((i[0]-x_c)**2 +(i[1]-y_c)**2)**0.5 for i in coordinates ] #list comprehension for euqiledian distances beetween freshly generated point and existing ones
            if not distances:
                coordinates.append((x_c, y_c))
                break
            elif min(distances)<ball_sep_dist:
                coordinates.append((x_c,y_c))
                break

    print(f'given c set: {coordinates}')
    phase_plates = [np.zeros(im_size,dtype=float) for i in range(len(coordinates))]

    im_out = np.zeros(im_size,dtype=float)

    for plate, coor in zip(phase_plates,coordinates):
        # todo split phase image into positive and negative
        print(coor)
        plate[coor]=random.choice([-1,1])
        plate = gaussian_filter(plate,next(sigma_generator))
        im_out += plate
        plt.imshow(plate)
        plt.title(coor)
        plt.colorbar()
        plt.show()

    plt.imshow(im_out)
    plt.title("merged")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":



    generate_ball()
