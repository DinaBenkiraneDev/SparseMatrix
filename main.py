import numpy as np
import matplotlib.pyplot as plt
from sparse import SparseMatrix, SparseTensor
import sys

# import notre fichier d'image
mnist_dataset = np.memmap('train-images-idx3-ubyte', offset=16, shape=(60000, 28, 28))


# fonction utiliser pour convertir notre tableau en triplet ou quadruplet
# methode qui convertit notre array de l'image en serie de tuplet
def array_to_tuple(array):
    tuple = []
    for b in range(0, len(array)):
        for c in range(0, len(array[0])):
            if array[b][c] != 0:
                tuple.append((b, c, array[b][c]))
    return tuple


# methode qui convertit notre array des images en serie de quadruplet
def array_to_quadruple(array):
    quadruple = []
    for image in range(0, len(array)):
        for line in range(0, len(array[0])):
            for column in range(0, len(array[0][0])):
                if array[image][line][column] != 0:
                    quadruple.append((image, line, column, array[image][line][column]))
    return quadruple


# Question 2 Variables utile pour la question.
# Permet de voir l'image a reproduire pour Sparse matrix
first_image = mnist_dataset[0].tolist()  # first_image est de taille (28,28)
# first_image mis dans SparseMatrix
first_image_matrix = SparseMatrix(array_to_tuple(first_image), (28, 28))
# Commande utilise pour visualiser first_image
# plt.imshow (first_image, cmap='gray_r')
# plt.show()

# On met tout le mnist_dataset en array
all_images = mnist_dataset.tolist()
# cree les quadruplet, pour all_images
all_images_sparsed = SparseTensor(array_to_quadruple(all_images), (60000, 28, 28))


# Verification des images
# Comparaison pixel par pixel de notre first_image.
def comparer_image1():
    for b in range(0, 28):
        for c in range(0, 28):
            # Regarde pour chaque valeur de tableau si les deux sont identiques.
            if first_image[b][c] != first_image_matrix.todense()[b][c]:
                print("Question 2:) Les deux images ne correspondent pas")
    print("Question 2:) Les deux images sont identiques")


# Compare chaque image a l'aide de fonction numpy, methode utilise pour la question 4b
def compare_images_matrix(array, n):
    for image in range(0, n):
        print("4B) progress image #", image,
              np.array_equal(array[image], SparseMatrix(array_to_tuple(array[image]), (28, 28)).todense()))


# Compare chaque image a l'aide de fonction numpy, methode utiliser pour la question 5a
def compare_images_tensor(array, array2, n):
    for image in range(0, n):
        print("5A) progress image #", image, ":", np.array_equal(array[image], array2[image]))

#5B)
def compare_espace_occupe(array, array2):
    print("Taille version dense:",sys.getsizeof(array),"bytes")
    print("Taille tenseur:",sys.getsizeof(array2),"bytes")


def main():
    # Question 2: Comparaison de la premiere image
    comparer_image1()

    # Question 4b
    compare_images_matrix(all_images, 60000)

    # Question 5a
    compare_images_tensor(all_images_sparsed.todense(), all_images, 60000)

    # Question 5b
    compare_espace_occupe(all_images,all_images_sparsed)

main()
