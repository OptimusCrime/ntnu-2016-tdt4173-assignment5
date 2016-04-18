

class ExtractStuff:

    def __init__(self):
        pass

    def extract(self, image):
        height = len(image)
        width = len(image[0])

        fragments = []
        for i in range(height - 20):
            for j in range(width - 20):
                fragments.append(
                    image[i:i + 20][j:j + 20]
                )

        return fragments