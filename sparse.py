import numpy as np
import matplotlib.pyplot as plt

class SparseMatrix:

    def __init__(self, fromiter, shape):
        n, m = shape
        self.n = n
        self.m = m
        self.nnz = len(fromiter)
        self.rowptr = self._to_row_ptr(fromiter)  # liste de taille n + 1 des intervalles des colonnes
        self.colind = self._to_col_ind(fromiter)  # liste de taille nnz des indices des valeurs non-nulles
        self.data = self._to_data(fromiter)  # liste de taille nnz des valeurs non-nulles

    # Fonction qui retourne la valeur correspondant a l'indice (i, j)
    def __getitem__(self, k):

        i, j = k
        # Index de debut du colind de la ligne i
        col_ind_index = self.rowptr[i]

        # Nombre d'elements non-nulls de la ligne actuelle
        nnz_curr_row = self.rowptr[i + 1] - self.rowptr[i]
        # Si il y a aucun elements non-nulls sur la ligne i
        if nnz_curr_row == 0:
            return 0

        #On fais une recherche dichotomique pour trouver notre valeur
        debut = col_ind_index
        fin = col_ind_index + nnz_curr_row
        trouve = False

        while debut<=fin and not trouve:
            #On fixe le point de milieu
            pt_milieu = (debut+fin)//2
            #Si la colonne recherchee est au milieu fixe de colind
            if self.colind[pt_milieu] == j:
                #On retourne la valeur correspondante a cet index dans data
                return self.data[pt_milieu]
            else:
                #Sinon si l'index est inferieur au milieu
                if j<self.colind[pt_milieu]:
                    fin = pt_milieu-1
                else:
                    #Sinon ca veut dire qu'il est au dessus
                    debut = pt_milieu+1
        return 0

    #Fonction qui encode la matrice en format dense
    def todense(self):
        # on initialise notre matrice avec les bonnes dimensions, rempli de 0
        # On ajoute une liste ayant comme taille le nombre de lignes et colonnes de la matrice
        matrice=[[0 for f in range(self.m)]for p in range(self.n)]

        #Ensuite pour tous les elements non-nuls
        #Cette variable va traverser le tableau colind selon la progression
        curr_col_ind_index = 0
        #On ajoute les elements non-nuls de la matrice ligne par ligne
        for d in range(0,len(self.rowptr)-1):
            #Nombre d'elements non-nulls de la ligne actuelle
            nnz_curr_row = self.rowptr[d+1]-self.rowptr[d]
            #On ajoute les elements non-nulls de la ligne selon la colonne
            for g in range(curr_col_ind_index,curr_col_ind_index+nnz_curr_row):
                matrice[d][self.colind[g]] = self.data[g]
                #Incrementation de l'index dans colind
                curr_col_ind_index+=1
        return matrice



    # Fonction qui retourne une liste de taille n + 1 des intervalles des colonnes
    def _to_row_ptr(self, fromiter):
        #Variable qui va donner le nombre de non-nulls dans chaque ligne
        _nbNnzLignes = [0]*self.n
        #Pour tous les triplets
        for d in range(0, self.nnz):
            #On ajoute un a la ligne correspondant au triplet en question
            _nbNnzLignes[fromiter[d][0]] += 1

        rowptr = list()
        # La liste doit toujours commencer par un 0
        rowptr.append(0)
        #Pour chaque liste
        for a in range(0, self.n):
            #On aditionne le nombre de non-nulls de cette ligne au rowptr precedant
            rowptr.append(_nbNnzLignes[a]+rowptr[a])
        return rowptr

    #Fonction qui retourne une liste de taille nnz des indices des valeurs non-nulles
    def _to_col_ind(self, fromiter):
        colind = []
        #Pour chaque triplet
        for d in range(0, self.nnz):
            #Ajouter la l'indice de colonne de la valeur non-nulle
            colind.append(fromiter[d][1])
        return colind

    #Fonction qui retourne une liste de taille nnz des valeurs non-nulles
    def _to_data(self, fromiter):
        data = []
        #Pour chaque triplet
        for d in range(0, self.nnz):
            #Ajouter la valeur non-nulle a data
            data.append(fromiter[d][2])
        return data


class SparseTensor:
    def __init__(self, fromiter, shape):
        n, m, z = shape
        self.n = n   #image
        self.m = m   #ligne
        self.z = z   #colonne
        self.nnz = len(fromiter)
        self.rowptr = self._to_row_ptr(fromiter)  # liste de taille n + 1 des intervalles des colonnes
        self.colind = self._to_col_ind(fromiter)  # liste de taille nnz des indices des valeurs non-nulles
        self.data = self._to_data(fromiter)  # liste de taille nnz des valeurs non-nulles


    #Fonction qui retourne la valeur correspondant a l'indice (i, j)
    def __getitem__(self, h):
        i, j, k = h

        # Index de debut du colind de la ligne i
        col_ind_index = self.rowptr[i][j]
        # Nombre d'elements non-nulls de la ligne actuelle
        nnz_curr_row = self.rowptr[i][j + 1] - self.rowptr[i][j]
        # Si il y a aucun elements non-nulls sur la ligne i
        if nnz_curr_row == 0:
            return 0

        # Recherche dichotomique, possible car colind est dans l'ordre
        debut = col_ind_index
        fin = col_ind_index + nnz_curr_row
        trouve = False

        while debut <= fin and not trouve:
            # On fixe le point de milieu
            pt_milieu = (debut + fin) // 2
            # Si la colonne recherchee est au milieu fixe de colind
            if self.colind[pt_milieu] == k:
                # On retourne la valeur correspondante a cet index dans data
                return self.data[pt_milieu]
            else:
                # Sinon si l'index est inferieur au milieu
                if k < self.colind[pt_milieu]:
                    fin = pt_milieu - 1
                else:
                    # Sinon ca veut dire qu'il est au dessus
                    debut = pt_milieu + 1
        return 0

    #Fonction qui encode la matrice en format dense
    def todense(self):

        # on initialise notre matrice avec les bonnes dimensions, rempli de 0
        # On ajoute une liste ayant comme taille le nombre i, lignes et colonnes de la matrice
        matrice=[[[0 for l in range(self.z)]for f in range(self.m)]for p in range(self.n)]

        # Ensuite pour tous les elements non-nuls
        # Cette variable va traverser le tableau colind selon la progression
        curr_col_ind_index = 0
        for i in range(0,self.n):
            #On ajoute les elements non-nuls de la matrice ligne par ligne
            for d in range(0, self.m):
                #Nombre d'elements non-nulls de la ligne actuelle
                nnz_curr_row = self.rowptr[i][d+1]-self.rowptr[i][d]
                #On ajoute les elements non-nulls de la ligne selon la colonne
                for g in range(curr_col_ind_index,curr_col_ind_index+nnz_curr_row):
                    matrice[i][d][self.colind[g]] = self.data[g]
                    #Incrementation de l'index dans colind
                    curr_col_ind_index+=1
        return matrice


    #Fonction qui retourne une liste de taille n + 1 des intervalles des colonnes
    def _to_row_ptr(self, fromiter):
        rowptr_final = []

        #Variable qui indexe la progression faite dans les quadruplets
        #(vu que leur index i est dans l'ordre)
        pointeur_quadruplets = 0

        for i in range(0, self.n):
            # Variable qui va donner le nombre de non-nulls dans chaque ligne
            nb_nnz_lignes = [0] * self.m
            #Variable qui contiendra tous les quadruplets ayant le meme i
            quadruplets_i = []

            #Jusqu'a temps qu'on arrive au premier index de i dans l'ensemble des quadruplets de fromiter
            while i>=fromiter[pointeur_quadruplets][0]:
                #Si le quadruplet a le meme premier index que i
                if fromiter[pointeur_quadruplets][0] == i:

                    #On l'ajoute dans quadruplets_i
                    quadruplets_i.append(fromiter[pointeur_quadruplets])
                #Si on est rendu au dernier index i
                if pointeur_quadruplets == self.nnz-1:
                    #On sort du while
                    break
                #Sinon on augmente le pointeur pour arriver au meme index que i
                #(ou un de plus grand, qui nous fera sortir de la boucle while)
                pointeur_quadruplets+=1


            #Tant qu'il y a des elements dans la liste de quadruplets
            while quadruplets_i:
                #On ajoute un a la ligne correspondant au quadruplet du dernier index
                nb_nnz_lignes[quadruplets_i[len(quadruplets_i)-1][1]] += 1
                #On pop le dernier pour ne pas avoir a le reverifier plus tard
                quadruplets_i.pop()

            rowptr = []
            #Si la liste est vide
            if not rowptr_final:
                # La liste doit toujours commencer par un 0
                rowptr.append(0)
            else:
                #Sinon on se base sur le rowptr d'index i precedent
                rowptr.append(rowptr_final[i-1][self.z])

            # Pour chaque liste
            for a in range(0, self.m):
                # On aditionne le nombre de non-nulls de cette ligne au rowptr precedant
                rowptr.append(nb_nnz_lignes[a] + rowptr[a])
            #On ajoute le row ptr de chaque i
            rowptr_final.append(rowptr)
        return rowptr_final

    #Fonction qui retourne une liste de taille nnz des indices des valeurs non-nulles
    def _to_col_ind(self, fromiter):
        colind = []
        #Pour chaque quadruplet
        for d in range(0,self.nnz):
            #Ajouter la l'indice de colonne de la valeur non-nulle
            colind.append(fromiter[d][2])
        return colind

    #Fonction qui retourne une liste de taille nnz des valeurs non-nulles
    def _to_data(self, fromiter):
        data = []
        #Pour chaque quadruplet
        for d in range(0, self.nnz):
            #Ajouter la valeur non-nulle a data
            data.append(fromiter[d][3])
        return data