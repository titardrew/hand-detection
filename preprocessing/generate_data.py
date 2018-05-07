"""This script was created to create new dataset for training and validation.

Note, that raw data dir has to be stuctured as follows:

    <dir>/
    |--- bad/
    |      +-- jpg images
    |--- left/
    |      +-- jpg images
    |--- screen/
    |      +-- jpg images (base)
    |--- right/
    |      +-- jpg images
    +--- test/
        +-- jpg images

Read docstrings for details.
"""

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from PIL import Image
from numpy import random
from os import listdir
from os import makedirs
from os.path import abspath
from os.path import isdir
from os.path import isfile
from os.path import join
from tqdm import tqdm


random.seed(0xFACE)

RAW_DIR_PATH = {'left' : '../raw_data/left/',
                'right' : '../raw_data/right/',
                'base' : '../raw_data/screen/',
                'bad' : '../raw_data/bad/'}
ROTATION_RATE = 10


def read_raw_img(kind):
    """Get generator to read certain raw data.

    Args:
        kind: 'left', 'right', 'bad' or 'base'

    Returns: generator.
    """

    mypath = RAW_DIR_PATH[kind]
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))
             and f[0] != '.']
    random.shuffle(files)

    if kind == 'bad':
        files *= 3

    for img in files:
        yield Image.open(mypath + img)


def count_raw_img(kind):
    """Get number of files in certain raw dataset.

    Args:
        kind: 'left', 'right', 'bad' or 'base'

    Returns: len(dataset)
    """

    mypath = RAW_DIR_PATH[kind]
    return len([f for f in listdir(mypath) if isfile(join(mypath, f))
                and f[0] != '.'])


def get_random_base():
    """Return random background image (from base dir)."""

    n_base = count_raw_img('base')
    img = "{}.jpg".format(random.randint(1, n_base + 1))
    return Image.open(RAW_DIR_PATH['base'] + img)


def place_entity(entity, base, x, y):
    """Place image into base.

    ! May change base !

    Args:
        entity: Image to place.
        base: Background image.
        x, y: Coordinates of top left corner of the image.
    """
    
    img = entity.copy().convert("RGBA")

    # Get random angle for placement
    angle = random.randint(-ROTATION_RATE, ROTATION_RATE)
    img = img.rotate(angle, expand=1)

    # Placement
    base.paste(img, (x, y), img)


def place_all(items, base):
    """ Place all items to base image.

    Args:
        items: Images to paste.
        base: Background image.

    """

    def intersect(x, y, pos, item_w, item_h, tol):
        for x2, y2 in pos:
            if (x <= x2 + item_w - tol) and (x2 <= x + item_w - tol) and \
               (y <= y2 + item_h - tol) and (y2 <= y + item_h - tol):
                return True
        return False


    positions = []
    new_base = base.copy()

    # Warning! Assumption that imgs are 189x110
    w, h = 189, 110
    # Warning! Assumption that items are 50x50
    item_w, item_h = 50, 50
    # Set tolerance for soft intersection
    tol = 10


    for item in items:
        # Patch: limit tries to 1000
        for _ in range(1000):
            x = random.randint(0, w - item_w * 0.8 + 1)
            y = random.randint(0, h - item_h * 0.8 + 1)

            if not intersect(x, y, positions, item_w, item_h, tol):
                break

        place_entity(item, new_base, x, y)
        positions.append((x, y))

    return new_base


def compose_img(source, destination, limit=0):
    """Generate dataset from raw data.

    New dataset is formed file by file in the next way:
    1. Choose random number (with custom probabilities) of raw images 50x50
       of each kind (bad, left and right).
    2. Place all picked elements into a background (which also is
       chosen randomly).

    Note:
    1. When new image is pasted to base one they it is placed without
       intersection (or with little one, which is controlled with `tol`)
    2. If nothing is picked, nothing will be produced.
    3. Image is usually pasted with random rotation.

    Args:
        source: Path to raw data.
        destination: Path to dir, where new set will be placed.
        limit: Max count of files in new dataset (if 0 than no limit).

    """

    # Check if dirs exist
    abs_path = abspath(source)
    if not isdir(abs_path):
        print("Source dir doesn't exist. Making (with -p)...")
        makedirs(abs_path)

    abs_path = abspath(destination)
    if not isdir(abs_path):
        print("Destination dir doesn't exist. Making (with -p)...")
        makedirs(abs_path)

    # Count raw images
    left_count = count_raw_img('left')
    right_count = count_raw_img('right')
    bad_count = count_raw_img('bad')

    # Get generator for reading.
    left_gen = read_raw_img('left')
    right_gen = read_raw_img('right')
    bad_gen = read_raw_img('bad')

    # Status bar.
    pbar = tqdm(total=left_count + right_count + bad_count)
    img_num = 1

    while left_count + right_count > 0:
        # Pick random count of left and right hands, 'bad' entities.
        picked_left = random.choice([0, 1, 2], p=[0.4, 0.55, 0.05])

        if picked_left == 0:
            picked_right = random.choice([0, 1, 2], p=[0.1, 0.7, 0.2])
        elif picked_left == 1:
            picked_right = random.choice([0, 1, 2], p=[0.4, 0.55, 0.05])
        else:
            picked_right = random.choice([0, 1], p=[0.42, 0.58])

        picked_bad = random.choice([0, 1], p=[0.15, 0.85]) 

        # Make sure that there're enough unused images left.
        picked_left = min(picked_left, left_count)
        picked_right = min(picked_right, right_count)
        picked_bad = min(picked_bad, bad_count)

        left_count -= picked_left
        right_count -= picked_right
        bad_count -= picked_bad

        # Place all items that were picked.
        if picked_bad + picked_right + picked_left > 0:
            base = place_all([next(left_gen)
                            for i in range(picked_left)] +
                            [next(right_gen)
                            for i in range(picked_right)] +
                            [next(bad_gen)
                            for i in range(picked_bad)],
                            get_random_base())

            pbar.update(picked_left + picked_right + picked_bad)

            # Saving the image.
            base.save("{}/{}.jpg".format(destination, img_num + int(1e6)))

            if img_num >= limit and limit > 0:
                break
            else:
                img_num += 1

    pbar.close()


def main(args):
    compose_img(source=args.source,
                destination=args.destination,
                limit=args.limit)
    print('Dataset has been generated successfully!')


if __name__ == '__main__':
    # Setting up the argument parser.
    parser = ArgumentParser(description='Generates dataset.',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', '--source', type=str, default="../data/",
                        help='Path to dir, where raw data is stored.')
    parser.add_argument('-d', '--destination', type=str, default="data",
                        help='Path to dir, where new data will be placed.')
    parser.add_argument('-l', '--limit', type=int, default=0,
                        help='Max size of generated set. (0 if no limit)')
    main(parser.parse_args())

