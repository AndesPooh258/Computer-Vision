import PIL
import random

class Padding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, img, **kwargs):
        # initialize result image with new size
        new_w = img.size[0] + self.padding * 2
        new_h = img.size[1] + self.padding * 2
        result = PIL.Image.new(mode=img.mode, size=(new_w, new_h), color=0)

        # paste the original image to the result image
        result.paste(img, box=(self.padding, self.padding))

        # return transformed result image
        return result

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, **kwargs):
        # determine starting point of cropping
        x = random.randint(0, img.size[0] - self.size)
        y = random.randint(0, img.size[1] - self.size)

        # crop the image based on the starting point
        result = img.crop(box=(x, y, x + self.size, y + self.size))

        # return transformed result image
        return result

class RandomFlip(object):
    # usually, applying random horizontal flip is better than random horizontal + vertical flip
    # vertical flip can be implemented similarly with img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, **kwargs):
        # determine whether to flip or not
        flip = random.random()

        # return transformed result image
        return img.transpose(PIL.Image.FLIP_LEFT_RIGHT) if flip >= self.p else img

class Cutout(object):
    def __init__(self, num_holes=1, size=8):
        self.num_holes = num_holes
        self.size = size
    
    def __call__(self, img, **kwargs):
        for _ in range(self.num_holes):
            # initialize a hole
            hole = PIL.Image.new(mode=img.mode, size=(self.size, self.size), color=0)

            # determine starting point of cutout
            x = random.randint(0, img.size[0] - self.size)
            y = random.randint(0, img.size[1] - self.size)

            # cutout on image
            img.paste(hole, box=(x, y))

        # return transformed result image
        return img
