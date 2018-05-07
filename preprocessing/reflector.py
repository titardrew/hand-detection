from PIL import Image
import os
from tqdm import tqdm
import xmltodict

def get_examples_list():
    dir_ = 'data'
    files = [f[:-4] for f in os.listdir(dir_)
             if os.path.isfile(os.path.join(dir_, f))
             and f[0] != '.']
    return files


def flip_image(img, vert=True):
    """Flip or mirror the image."""
    if vert:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return img.transpose(Image.FLIP_LEFT_RIGHT)


def mirror_vertical_save():
    files = get_examples_list()
    for filename in tqdm(files):
        path = 'labels/' + filename + ".xml"
        with open(path) as xml:
            d = xmltodict.parse(xml.read())
            d['annotation']['filename'] = filename + '_v.jpg'
            d['annotation']['path'] = d['annotation']['path'][:-4] + '_v.jpg'
            if 'object' not in d['annotation']:
                it = []
            elif type(d['annotation']['object']) != list:
                it = [d['annotation']['object']]
            else:
                it = d['annotation']['object']

            for y in it:
                x_min = int(y['bndbox']['xmin'])
                x_max = int(y['bndbox']['xmax'])
                y_min = int(y['bndbox']['ymin'])
                y_max = int(y['bndbox']['ymax'])

                if y['name'] == 'left':
                    y['name'] = 'right'
                else:
                    y['name'] = 'left'

                # y['bndbox']['xmin'] = str(x_max)
                y['bndbox']['ymin'] = str(110 - y_max + 1)
                # y['bndbox']['xmax'] = str(x_min)
                y['bndbox']['ymax'] = str(110 - y_min + 1)

            new = '\n'.join(xmltodict.unparse(d, pretty=True).split("\n")[1:])
            with open('labels/' + filename + '_v.xml', 'w') as f:
                f.write(new)

        im = Image.open("data/" + filename + ".jpg")
        flip_image(im).save("data/" + filename + "_v.jpg")


def mirror_horizontal_save():
    files = get_examples_list()
    for filename in tqdm(files):
        path = 'labels/' + filename + ".xml"
        with open(path) as xml:
            d = xmltodict.parse(xml.read())
            d['annotation']['filename'] = filename + '_h.jpg'
            d['annotation']['path'] = d['annotation']['path'][:-4] + '_h.jpg'
            if 'object' not in d['annotation']:
                it = []
            elif type(d['annotation']['object']) != list:
                it = [d['annotation']['object']]
            else:
                it = d['annotation']['object']

            for y in it:
                x_min = int(y['bndbox']['xmin'])
                x_max = int(y['bndbox']['xmax'])
                y_min = int(y['bndbox']['ymin'])
                y_max = int(y['bndbox']['ymax'])

                if y['name'] == 'left':
                    y['name'] = 'right'
                else:
                    y['name'] = 'left'

                y['bndbox']['xmin'] = str(189 - x_max + 1)
                # y['bndbox']['ymin'] = str(y_max)
                y['bndbox']['xmax'] = str(189 - x_min + 1)
                # y['bndbox']['ymax'] = str(y_min)

            new = '\n'.join(xmltodict.unparse(d, pretty=True).split("\n")[1:])
            with open('labels/' + filename + '_h.xml', 'w') as f:
                f.write(new)

        im = Image.open("data/" + filename + ".jpg")
        flip_image(im, vert=False).save("data/" + filename + "_h.jpg")

if __name__=="__main__":
    mirror_vertical_save()
    mirror_horizontal_save()
