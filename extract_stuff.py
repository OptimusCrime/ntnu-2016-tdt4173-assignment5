import numpy as np


class ExtractStuff:

    def __init__(self):
        pass

    def extract(self, image):
        height = len(image)
        width = len(image[0])

        locations = []
        fragments = []
        for i in range(0, height - 20, 5):
            for j in range(0, width - 20, 5):
                temp_fragments = image[i:(i + 20), j:(j + 20)].reshape((400, ))
                #print(temp_fragments)

                if np.count_nonzero(temp_fragments) > 0:
                    fragments.append(temp_fragments)
                    locations.append((j, i))
        return np.array(fragments), locations