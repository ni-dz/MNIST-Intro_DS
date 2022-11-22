
import gzip

import numpy

#MNIST-Daten aus dem Trainingssatz der label auslesen
def get_labels(location):
    with gzip.open(location) as file:
        first_numbers = int.from_bytes(file.read(8), 'big') #Auslesen der magic number und der number of items (jeweils 32 bit integer)
        labels = file.read() #auslesen der bits danach und abspeichern in labels
        labels = numpy.frombuffer(labels, dtype=numpy.int8) #Abspeichern der labels in einem Array. Jedes Label ist in einem byte abgespeichert, weshalb als Datentyp int8 gewählt werden muss.
        return labels


def get_images(location):
    with gzip.open(location) as file:
        first_numbers = int.from_bytes(file.read(16), 'big')  # Auslesen der magic number und der number of items (jeweils 32 bit integer)

        images = file.read()
        images = numpy.frombuffer(images, dtype=numpy.uint8)

        twod = [[None for _ in range(28)] for _ in range(28)]
        threed = [twod for _ in range(60000)]

        for x in range(60000):
            for y in range(28):
                for z in range(28):
                    try:
                        threed[x][z][y] = images[(x+1)*(y+1)*(z+1)]
                    except IndexError:
                        pass



    return threed
