#ce fichier a pour but de faire un tirage aléatoire d un
#découpage selon le Recom avec algo de wilson pour le tirage du spanning tree



#<>

#pour les histogrammes
import matplotlib.pyplot as plt  # Module pour tracer les graphiques
import numpy as np
import pandas as pd
import seaborn as sns

import copy, math
import time
import sys
import random
import signal

##################################
##################################
####### exception pour le timeout #

TEMPS_LIMITE = 8*60

def recuperation(signum, frame):
    raise Exception("timeout !")

###################################
###################################




####### Lecture fichier ###########
###################################
# print("Département :  ")
# numDep = input()
# print("fichier d'entrée pour créer le graphe (de la forme graphe_depts/numéro.out) :  ")
# nomFichierEntree= input()
# print("fichier d'entrée pour prendre le découpage de départ (de la forme xpour100/sortienuméro.txt) :  ")
# nomFichierDecoupInit = input()
# ## Nombre de circonscriptions
# print("nombre de circonscription dans le département :  ")
# nci=int(input())
# ## Taux d'ecart max a la moyenne (population)
# print("taux d'écart à la moyenne (population)")
# taux = float(input())

# #gestion de la Corse <>
# if(numDep == '20'):
#     numDep = '2A'
#     nomFichierEntree = "graphe_depts/2A.out"
# elif(numDep == '96'):
#     numDep = '2B'
#     nomFichierEntree = "graphe_depts/2B.out"

# #Paris et Lyon sont gérés à part 

# if(numDep == '75' or numDep == '69'):
#     sys.exit(1)
# nomFichierSortie= "avec_dist_recom/sortie_Recom" + numDep + ".txt"







###################################
def lecture():

    mon_fichier = open(nomFichierEntree, "r")
    contenu = mon_fichier.readlines()
    mon_fichier.close()

    tableau=[]
    #i=0
    for ligne in contenu:
        ligne=ligne.replace(',',' ')
        tableau.append(ligne.split())
    #print(tableau)




    nbCantons=int(tableau[0][-1])
    #print(nbCantons)
    Lc=[]
    for i in range(1,nbCantons+1):
        elt = tableau[i][:]
        nomCanton = elt[1]
        for i in range(2,len(elt)-2): #Gestion des cantons avec noms à plusieurs mots (ex : "La Flèche")
            nomCanton += " "
            nomCanton += elt[i]
        elt2 = [int(elt[0])-1,nomCanton,int(elt[-2]),elt[-1]]
        Lc.append(elt2)
        """Lc[i-1][0]=int(Lc[i-1][0])-1 ## Decalage numerotation dans les fichiers
        Lc[i-1][2]=int(Lc[i-1][2])"""
        #Lc[i-1][3]=int(Lc[i-1][3])
    t=len(tableau)
    Ls=[]
    for i in range(nbCantons+2,t):
        li=tableau[i][:]
        for j in range(len(li)):
            if (j<4):#coordonnees des segments
                li[j]=float(li[j]) 
            if (j==4 or j==5): # cantons adjacents par ce segment
                li[j]=int(li[j]) - 1
		#longueur=math.sqrt((li[0]-li[2])*(li[0]-li[2])+(li[1]-li[3])*(li[1]-li[3]))
        R=6400 #rayon approximatif de la terre
        dx=R*math.cos(2*math.pi*li[1]/360)*2*math.pi*(li[0]-li[2])/360
        dy=R*2*math.pi*(li[1]-li[3])/360
        longueur=math.sqrt(dx*dx+dy*dy) ## Certainement un arccosinus pour etre exact
        li.append(longueur)
        Ls.append(li)

    # lecture pour récupérer la decomposition initiale
    fichier_initial = open(nomFichierDecoupInit, "r")
    contenu_initial = fichier_initial.readlines()

    fichier_initial.close()

    # print(nbCantons)
    # print("voila contenu initial :  ")
    # print(contenu_initial)
    # print("fin contenu_initial")

    ligne_interessante = contenu_initial[2]
    decoupage_reel = (ligne_interessante.split()[:nbCantons])
    #print(decoupage_reel)
    decoup = []
    for elt in decoupage_reel:
        decoup.append(int(elt))
    #print(decoup)
    if len(decoup) != nbCantons:
        print("ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        raise ZeroDivisionError
    return ([nbCantons,Lc,len(Ls),Ls, decoupage_reel])



# la normalisation n'est pas incroyable : on doit faire une deuxième fct
#on met les params pour de lautomatisation pour calculer taille coupe et nb arbrs couvrant et faire les graphiques entre les deux
def lecture_bis(): #nomFichierEntree, nomFichierDecoupInit):

    mon_fichier = open(nomFichierEntree, "r")
    contenu = mon_fichier.readlines()
    mon_fichier.close()
    print("dans lecture_bis")
    print(nomFichierEntree)

    tableau=[]
    #i=0
    for ligne in contenu:
        ligne=ligne.replace(',',' ')
        tableau.append(ligne.split())
    #print(tableau)




    nbCantons=int(tableau[0][-1])
    print(tableau[0])
    print(tableau[1])
    #print(nbCantons)
    Lc=[]
    for i in range(1,nbCantons+1):
        elt = tableau[i][:]
        nomCanton = elt[1]
        for i in range(2,len(elt)-2): #Gestion des cantons avec noms à plusieurs mots (ex : "La Flèche")
            nomCanton += " "
            nomCanton += elt[i]
        elt2 = [int(elt[0])-1,nomCanton,int(elt[-2]),elt[-1]]
        Lc.append(elt2)
        """Lc[i-1][0]=int(Lc[i-1][0])-1 ## Decalage numerotation dans les fichiers
        Lc[i-1][2]=int(Lc[i-1][2])"""
        #Lc[i-1][3]=int(Lc[i-1][3])
    t=len(tableau)
    Ls=[]
    for i in range(nbCantons+2,t):
        li=tableau[i][:]
        for j in range(len(li)):
            if (j<4):#coordonnees des segments
                li[j]=float(li[j]) 
            if (j==4 or j==5): # cantons adjacents par ce segment
                li[j]=int(li[j]) - 1
		#longueur=math.sqrt((li[0]-li[2])*(li[0]-li[2])+(li[1]-li[3])*(li[1]-li[3]))
        R=6400 #rayon approximatif de la terre
        dx=R*math.cos(2*math.pi*li[1]/360)*2*math.pi*(li[0]-li[2])/360
        dy=R*2*math.pi*(li[1]-li[3])/360
        longueur=math.sqrt(dx*dx+dy*dy) ## Certainement un arccosinus pour etre exact
        li.append(longueur)
        Ls.append(li)

    # lecture pour récupérer la decomposition initiale
    fichier_initial = open(nomFichierDecoupInit, "r")
    contenu_initial = fichier_initial.readlines()

    fichier_initial.close()

    # print(nbCantons)
    # print("voila contenu initial :  ")
    # print(contenu_initial)
    # print("fin contenu_initial")
    

    #pour récupérer tous les découpages : 
    tous_les_decoupages = []
    # print(contenu_initial[1].split())
    nombre_de_trucs_a_enlever = int((contenu_initial[1].split())[0])
    toutes_les_lignes_int = contenu_initial[2:]
    for i in range(len(toutes_les_lignes_int)):
        decoupage = toutes_les_lignes_int[i].split()[nombre_de_trucs_a_enlever:]
        decoup_bien_forme = []
        for elt in decoupage:
            decoup_bien_forme.append(int(elt))
        print("decoupage : ")
        print(decoup_bien_forme)
        if len(decoupage) != nbCantons:
            print("ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh" + str(len(decoupage)))
            #raise ZeroDivisionError
        tous_les_decoupages.append(decoup_bien_forme)
        
        

    return ([nbCantons,Lc,len(Ls),Ls, tous_les_decoupages])


def calcule_k_1_et_k2(numDep): #enfin, seulement k_2 pour l'instant -> finalement non c'est fait (19 juin)
    #finalement on calcule k_1 mais k_2 c'est n'importe quoi pour l'instant (20 juin)

    fichier_coos = "graphe_depts/" + str(numDep) + ".out"
    print(fichier_coos)
    nomFichierEntree = fichier_coos

    mon_fichier = open(nomFichierEntree, "r")
    contenu = mon_fichier.readlines()
    mon_fichier.close()

    tableau=[]
    #i=0
    for ligne in contenu:
        ligne=ligne.replace(',',' ')
        tableau.append(ligne.split())
    #print(tableau)




    nbCantons=int(tableau[0][-1])
    print("dans la focntion_k_1_k_2 !!")
    print(tableau[0])
    print(tableau[1])
    #print(nbCantons)
    Lc=[]
    for i in range(1,nbCantons+1):
        elt = tableau[i][:]
        nomCanton = elt[1]
        for i in range(2,len(elt)-2): #Gestion des cantons avec noms à plusieurs mots (ex : "La Flèche")
            nomCanton += " "
            nomCanton += elt[i]
        elt2 = [int(elt[0])-1,nomCanton,int(elt[-2]),elt[-1]]
        Lc.append(elt2)
        """Lc[i-1][0]=int(Lc[i-1][0])-1 ## Decalage numerotation dans les fichiers
        Lc[i-1][2]=int(Lc[i-1][2])"""
        #Lc[i-1][3]=int(Lc[i-1][3])
    t=len(tableau)
    Ls=[]
    for i in range(nbCantons+2,t):
        li=tableau[i][:]
        for j in range(len(li)):
            if (j<4):#coordonnees des segments
                li[j]=float(li[j]) 
            if (j==4 or j==5): # cantons adjacents par ce segment
                li[j]=int(li[j]) - 1
		#longueur=math.sqrt((li[0]-li[2])*(li[0]-li[2])+(li[1]-li[3])*(li[1]-li[3]))
        R=6400 #rayon approximatif de la terre
        dx=R*math.cos(2*math.pi*li[1]/360)*2*math.pi*(li[0]-li[2])/360
        dy=R*2*math.pi*(li[1]-li[3])/360
        longueur=math.sqrt(dx*dx+dy*dy) ## Certainement un arccosinus pour etre exact
        li.append(longueur)
        Ls.append(li)


    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)
    print(liste_adj)

    if len(liste_adj) == 0 : 
        return (0,0)
    k_1 = len(liste_adj[0])
    k_1_vrai = len(liste_adj[1])
    if k_1 < k_1_vrai : 
        k_1 = k_1_vrai
        k_1_vrai = len(liste_adj[0])
    for i in range(2, len(liste_adj)):
        if len(liste_adj[i]) >= k_1:
            k_1_vrai =  k_1
            k_1 = len(liste_adj[i])
        elif len(liste_adj[i]) >= k_1_vrai:
            k_1_vrai = len(liste_adj[i])


    # ~~~~~~~~~~~ calcul de k_2 ~~~~~~~~~~~~~~~~~~

    
    fichier_coos = "graphe_depts/" + str(numDep) + ".out"
    fichier = open(fichier_coos, "r")
    lignes = fichier.readlines()
    fichier.close()



    nb_dechets = int(lignes[0].split()[-1])
    f = lignes[nb_dechets+2:]
    #frontieres contient les coos des segments qui forment les frontières

    dico_intersections = {}

    precedent = (-1, -1)
    pt_prec = (-1, -1)
    for i in range(len(f)):
        frontieres = f[i].split()
        if frontieres[-2] != "0" and frontieres[-1] != "0" : 
            # print(precedent)
            if (frontieres[-2], frontieres[-1]) != precedent : 
                pt_en_c = (frontieres[0], frontieres[1])
                if not (pt_en_c in dico_intersections ):
                    dico_intersections[pt_en_c] = 1
                else : 
                    dico_intersections[pt_en_c] += 1

                if not (pt_prec in dico_intersections ):
                    dico_intersections[pt_prec] = 1
                else : 
                    dico_intersections[pt_prec] += 1
            pt_prec = (frontieres[2], frontieres[3])
            precedent = (frontieres[-2], frontieres[-1])

    print("dico intersectionnnn ")
    print(dico_intersections)


    # NB : on doit diviser par deux car chaque arête est comptée une fois par extrémité : ie deux fois !
    maxii = 0
    k_2 = 0
    for elt in dico_intersections.keys():
        if dico_intersections[elt]//2 >= maxii : 
            k_2 = maxii 
            maxii = dico_intersections[elt]//2
        elif dico_intersections[elt]//2 > k_2:
            k_2 = dico_intersections[elt]//2


    print("eeeeee")
    return (k_1_vrai, k_2)


################################################
## Methodes de type get
####################################

def popu(cant):
    return cant[2]
def taille(cant):
    return cant[3]

def cant1(seg):
    return seg[4]
def cant2(seg):
    return seg[5]
def long(seg):
    return seg[6]


###############################################
## Graphe d'adjacence
################################################

## Renvoie la matrice d'adjacence du graphe des cantons
def graphadj(nc,Lc,ns,Ls):
    print("##############################################")
    print(Ls)
    mat=[]
    for i in range(nc):
        mat.append([0]*nc)
    for seg in Ls:
        if (cant2(seg)>0): ##le canton n'est pas en limite du departement
            mat[cant1(seg)][cant2(seg)]=1
            mat[cant2(seg)][cant1(seg)]=1
    print(mat)
    return mat

### Renvoie la liste d'adjacence du graphe
def listeadj(mat):
    n = len(mat)
    ls = [[i for i in range(n) if mat[j][i]==1] for j in range(n)]
    #for l in ls:
    #    print(l)
    return ls



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# 
# 
# ### Ma partie réellement #####

# On considère que les cantons sont ordonés de 0 à len(mat) = nc (respo pour la liste d'aj et la mat d'adj)
# On considère qu'un découpage est un tableau de taille nc donc avec -1 si non assigné (n'arrivera jamais a priori), i plus grand que 0 ssi dans le disctrict i
# 
#
# V1: utilise listes d'ajacence
#
#
# Que prendre comme découpage initial ?
# -possibilité de prendre un découpage aléatoire construit itérativement (et deterministiquement)
# -possibilité de faire un truc aléatoire mais pas forcément connexe : fonctionnerait ?
# -possibilité de prendre le x-ème (toujours le même) construit itérativement (et déterministiquement)
# -


# Subalgos utilisés : 
# Algo de Wilson pour trouver un arbre couvrant aléatoire (uniforme)
# random successeur 

#liste adj : liste adjecance du graphe, part en c : tableau de taille nc qui assigne chaque canton a sa circo en cours
# circ1,2 : circo à fusionner et spliter
# s : sommet dont on doit trouver un successeur random
#NB : on est dand un graphe non orienté
def random_successeur(liste_adj, part_en_c, circ1, circ2, s):
    # print("noeud considéré dans random successeur : " + str(s))
    # print(part_en_c)
    #on prend direct un voisin de s
    random_node = liste_adj[s][random.randint(0,len(liste_adj[s])-1)]
    # print(liste_adj[s])

    #vérifier la condition ici
    # print("circo 1 : " + str(circ1) + "  circo 2 : " + str(circ2))
    while((part_en_c[random_node] != circ1 and part_en_c[random_node] != circ2) or random_node == s): 
       # print(random_node)
        random_node = liste_adj[s][random.randint(0,len(liste_adj[s])-1)]


    return random_node

# on recode une union find rapidement
#convention : on aura un tableau  des noeuds tq val = -2 si n'est pas dans l'uf (conforme avec ce qu'on faisait da wilson)
# tab[noeud] = (noeud, prof) ssi noeud et la racine de son arbre et prof est la prof max 
# sinon tab[noeud] = (parent, prof) de noeud, prof n'a aucun sens
def uf_union(uf, noeud1, noeud2):
    uf[noeud1] = noeud2
    return uf 

#on ne fait pas la complétion des chemins pour avoir un arbre
def uf_find(uf, noeud):
    if uf[noeud] == noeud :
        return noeud 
    else :
        return  uf_find(uf, uf[noeud])
        

    

#fonction qui a pour but de générer des arbres couvrant aléatoires (non uniformes mais ça peut etre cool quand même)
# choisit des poids aléatoires et applique kruskal dessus 
def kruskal_poids_alea(liste_adj, part_en_c, circ1, circ2):
    # génère l'uf :
    uf = [-2] * len(liste_adj)
    for i in range(len(part_en_c)):
        if part_en_c[i] == circ1 or part_en_c[i] == circ2 :
            uf[i] = i 

    #genere les poids_aléatoires 
    #on considère qu'il n,'y aura pas de collision vu la petite taille de nos graphes !
    dico = {}

    #i est un noeud dugraphe si je comprends bien
    for i in range(len(part_en_c)):
        #si le noeud doit ête mis dans l'arbre
        if uf[i] != -2:
            for elt in liste_adj[i]:
                #si le voisin doit aussi être dans l'arbre ie que l'arete entre eux doit aussi etre dans l'arbre
                if uf[elt] != -2 :
                    poids_alea = random.random()
                    #on met les deux extrémités de l'arete dans le dico
                    dico[poids_alea] = (i, elt)

    poids_tries = sorted(dico.keys())

    #fait l'uf : 
    # on prend une par une toutes les aretes concernées en les associant à leur poids
    for poids in poids_tries:
        elt = dico[poids] #c'est un couple de deux noeuds qui sont les extréités de l'arete

        if uf_find(uf,elt[0]) != uf_find(uf,elt[1]):
            # print("on unifie")
            uf = uf_union(uf,uf_find(uf,elt[0]), uf_find(uf,elt[1]))
            # print(uf)
    assert(len(poids_tries) > 0)


    
    #on fait la bonne convention pour la racine
    # racine = uf_find(dico[poids_tries[0]])
    #on peut mettre n'importe quel noeud comme racine non ?
    #dans les faits mieux de mettre la racine de l'uf (ie celui qui s'a comme père) sinon on a une boucle
    uf[uf_find(uf, dico[poids_tries[0]][0])] = -1 
    return uf
    

#liste adj : liste adjecance du graphe, part en c : tableau de taille nc qui assigne chaque canton a sa circo en cours
# circ1,2 : circo à fusionner et spliter
#retourne un arbre sous forme de tableau qui pointe vers prédecesseur (ie tous pointent plus proche de la racine)
def algo_de_wilson(liste_adj, part_en_c, circ1, circ2, Lc):

    poids_de_l_arbre = 0
    # print(circ1)
    # print(circ2)
    # print("on arrive voir Wilson")
    #on tire la racine aléatoire : 
    #pour avoir uniforme on fait ce truc pas fou :
    nb_cantons = len(liste_adj)

    random_root = random.randint(0,nb_cantons-1)
    while(part_en_c[random_root] != circ1 and part_en_c[random_root] != circ2): 
        random_root = random.randint(0,nb_cantons-1)
    
    # print("Wilson a la root aléatoire")

    #convention :
    # -1 : n'est pas du tout dans le spanning tree
    # 1 : n'est pas dans le st mais va y etre bientot
    # 0 est dans le st
    Previous = [] #on fait des listes de listes car un elt peut avoir plusieurs enfants 
    for i in range(nb_cantons):
        Previous.append([])
    InTree = [-1] * nb_cantons
    for i in range(nb_cantons):
        if part_en_c[i] == circ1 or part_en_c[i] == circ2 : 
            InTree[i] = 1
    #convention : 
    # -2 : pour noeud qui ne sont pas dans st (peuvent être ajoutés unjour ou non)
    # -1 : pour la rac du st
    # x différent : prédec dans le st 
    Next = [-2] * nb_cantons
    Next[random_root] = -1
    InTree[random_root] = 0
    for i in range (nb_cantons):
        
        if InTree[i] != -1 :
            
            u = i
            while not (InTree[u] == 0):
                # print("on a u éqal à   " + str(u))
                # print("c1 et c2 : " + str(circ1) + "  " + str(circ2))
                Next[u] = random_successeur(liste_adj, part_en_c, circ1, circ2, u)

                u = Next[u]
                assert(InTree[u] != -1)
            u = i 
            while not (InTree[u] == 0):
                poids_de_l_arbre += popu(Lc[u])
                

                Previous[Next[u]].append(u)
                InTree[u] = 0
                u = Next[u]
                
    return Next, Previous, poids_de_l_arbre, random_root

def partition_en_deux_arbres(liste_adj, part_en_c, c1, c2, node, T):
    #on prend la convention : 0 coté racine de l'arête retirée, 1 sinon
    # print("on est dans partition en deux arbres")
    # print(T)
    # print(node)
    # print(T[node])


    changement = True
    res = [-1] * len(T)

    res[node] = 1
    res[T[node]] = 0

    nn = T[node]
    while T[nn] != -1 and T[nn] != -2 :
        res[T[nn]] = 0
        nn = T[nn]
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    while changement: 
    
        #print("nouveau tour de boucle")
        # print(T)
        # print(res)
        changement = False
        for i in range(len(T)): 

            #avec ça tout ce qui est dans les sous-arbres des extrémités de l'arête supprimée vont etre marqués

            if T[i] != -2 and res[i] == -1:
                if res[T[i]] != -1: 
                    #print("on modifie res de : " + str(i))
                    res[i] = res[T[i]]
                    
                    changement = True


    # On renvoie dexu tableaux de booléens qui représentent l'appartenance à Ti
    T1 = [False] * len(T)
    T2 = [False] * len(T)

    for i in range(len(T)): 
        if res[i] == 0 :
            T1[i] = True
        if res[i] == 1:
            T2[i] = True
    return T1,T2


def fonction_simple(decoup_en_c, circ1, circ2):
    T1 = [False] * len(decoup_en_c)
    T2 = [False] * len(decoup_en_c)
    for i in range(len(decoup_en_c)):
        if decoup_en_c[i] == circ1:
            T1[i] = True
        if decoup_en_c[i] == circ2:
            T2[i] = True 
    return T1, T2


# T est une liste de booléen tq T[i] dit si i est dans l'arbre ou non
# Lc est le résultat de la lécture du fichier json (qui est converti en (liste/mat) d'adj par la suite)
def taille_arbre(T, Lc): 
    res = 0
    for i in range (len(T)):
        if T[i]:
            res += popu(Lc[i])
    return res

def calcule_tous_les_poids_de_maniere_utile(parent, enfants, Lc, racine, tab_poids):
    if enfants[racine] == []:
        tab_poids[racine] = popu(Lc[racine])
    else : 
        somme = 0
        for elt in enfants[racine]:
            tab_poids = calcule_tous_les_poids_de_maniere_utile(parent, enfants, Lc, elt, tab_poids)
            somme += tab_poids[elt]
        tab_poids[racine] = somme + popu(Lc[racine])
    return tab_poids

        
#le principe est que précédent est égal soit à noeud en bas soit à parent(noeud en bas), 
# si précédent est égal à noeud en bas alors on va se décaler vers le haut (au sens de la focntion de la première partie).
# si précédent est égal à parent(noeud en bas) alors on va se décaler vers le bas (au sens de la fonction de lapremière partie).
# cas à part, si on vient d'arriver dans la fonction de deuxième étape, alors présédent vaudra -444 qui n'est pas un noeud valide, donc on ira en haut ET en bas ! 
def fonction_recursive_deuxieme_etape(parent, enfants, Lc, borne_inf, borne_sup, noeud_en_bas, precedent, poids_en_bas, poids_en_haut, poids_arbre):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    res = []

    if not(borne_inf <= poids_en_bas <= borne_sup) or not(borne_inf <= poids_en_haut <= borne_sup):
        return []
    
    else : 
        res.append(noeud_en_bas)
        #cas o`u on décale vers le bas (pas incompatible avec le fait de se déplacer vers le haut)
        if noeud_en_bas != precedent:
            for elt in enfants[noeud_en_bas]:
                res += fonction_recursive_deuxieme_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, noeud_en_bas, poids_arbre[elt], poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre)


        #cas du décalage vers le haut
        if parent[noeud_en_bas] != precedent :
            noeud_bis = parent[noeud_en_bas]
            #on recalcule les poids : ça consiste essentiellement en déplacer noeud_bis de poisd en bas vers poids en haut

            if parent[noeud_bis] != -1 : # on vérifie qu'on ne va pas appeler sur la racine
            #problème si noeud bis devient le noeud en haut, il faut aller voir s'il y a des enfants à aller voir ! 
                

                poids_pour_passer_en_param_plus_tard = poids_en_haut

                for elt in enfants[noeud_bis]:
                    poids_pour_passer_en_param_plus_tard -= poids_arbre[elt]
                    res += fonction_recursive_deuxieme_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, noeud_bis, poids_arbre[elt], poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre)

                poids_pour_passer_en_param_plus_tard -= popu(Lc[noeud_bis])
                res += fonction_recursive_deuxieme_etape(parent, enfants,Lc, borne_inf, borne_sup, noeud_bis,noeud_bis, poids_en_bas + poids_en_haut - poids_pour_passer_en_param_plus_tard, poids_pour_passer_en_param_plus_tard, poids_arbre )
            else : 
                for elt in enfants[noeud_bis]:
                    res += fonction_recursive_deuxieme_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, poids_arbre[elt], noeud_bis, poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre)
    return res




        



#premiere étape est pour dire si on est dans la premiere etape ou non 
#cf fonction deuxieme etape pour les conventions sur precedent
def fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, noeud_en_bas, precedent, poids_en_bas, poids_en_haut,poids_arbre, premiere_etape : bool):
    if noeud_en_bas == -2 or parent[noeud_en_bas] == -2 : 
        return None
    print("on fait un appel à la fonction recursive premiere etape yeaaaaah")
    print("arete_en_c : "+ str(noeud_en_bas))
    # vérifier si noeuds en bas n'est pas la racine peut être  ? -> non c'est bon c'est fait au moment de faire la récursion
    print(poids_en_bas)
    print(poids_en_haut)
    print(borne_inf)
    print(borne_sup)
    if premiere_etape : 
        if poids_en_bas <borne_inf : 
            if poids_en_haut < borne_inf : 
                raise ZeroDivisionError
            else : 
                noeud_bis = parent[noeud_en_bas]
                #on recalcule les poids : ça consiste essentiellement en déplacer noeud_bis de poisd en bas vers poids en haut
                if parent[noeud_en_bas] != precedent:
                    if parent[noeud_bis] != -1 : # on vérifie qu'on ne va pas appeler sur la racine
                    #problème si noeud bis devient le noeud en haut, il faut aller voir s'il y a des enfants à aller voir ! 
                        

                        poids_pour_passer_en_param_plus_tard = poids_en_haut

                        for elt in enfants[noeud_bis]:
                            if elt != noeud_en_bas:
                                poids_pour_passer_en_param_plus_tard -= poids_arbre[elt]
                                ee =  fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, noeud_bis, poids_arbre[elt], poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre, premiere_etape)
                                if ee != None : 
                                    return ee
                        poids_pour_passer_en_param_plus_tard -= popu(Lc[noeud_bis])
                        return fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, noeud_bis, noeud_bis, poids_en_bas + poids_en_haut - poids_pour_passer_en_param_plus_tard, poids_pour_passer_en_param_plus_tard, poids_arbre, premiere_etape )
                    else : 
                        for elt in enfants[noeud_bis]:
                            
                            ee = fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, noeud_bis, poids_arbre[elt], poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre, premiere_etape)
                            if ee != None : 
                                return ee
        elif poids_en_bas > borne_sup : 
            if poids_en_haut > borne_sup : 
                raise ZeroDivisionError
            elif precedent != noeud_en_bas : 
                if elt != noeud_en_bas:

                    for elt in enfants[noeud_en_bas]:
                        ee = fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, poids_arbre[elt], noeud_en_bas, poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre, premiere_etape)
                        if ee != None : 
                            return ee


                
        else : 
            if poids_en_haut > borne_sup : 
                noeud_bis = parent[noeud_en_bas]
                #on recalcule les poids : ça consiste essentiellement en déplacer noeud_bis de poisd en bas vers poids en haut
                if precedent != parent[noeud_en_bas]:
                    if parent[noeud_bis] != -1 : # on vérifie qu'on ne va pas appeler sur la racine
                    #problème si noeud bis devient le noeud en haut, il faut aller voir s'il y a des enfants à aller voir ! 
                        

                        poids_pour_passer_en_param_plus_tard = poids_en_haut
                        if precedent != noeud_en_bas:

                            for elt in enfants[noeud_bis]:
                                poids_pour_passer_en_param_plus_tard -= poids_arbre[elt]
                                ee = fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, noeud_bis, poids_arbre[elt], poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre, premiere_etape)
                                if ee != None : 
                                    return ee

                        poids_pour_passer_en_param_plus_tard -= popu(Lc[noeud_bis])
                        return fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, noeud_bis, noeud_bis, poids_en_bas + poids_en_haut - poids_pour_passer_en_param_plus_tard, poids_pour_passer_en_param_plus_tard, poids_arbre, premiere_etape )
                    else : 
                        for elt in enfants[noeud_bis]:
                            ee = fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, noeud_bis, poids_arbre[elt], poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre, premiere_etape)
                            if ee != None : 
                                return ee
            elif poids_en_haut < borne_inf:
                if precedent != noeud_en_bas:
                    for elt in enfants[noeud_en_bas]:
                        ee = fonction_recursive_premiere_etape(parent, enfants,Lc, borne_inf, borne_sup, elt, noeud_en_bas, poids_arbre[elt], poids_en_bas + poids_en_haut - poids_arbre[elt], poids_arbre, premiere_etape)
                        if ee != None : 
                            return ee
            else : 
                return fonction_recursive_deuxieme_etape(parent, enfants, Lc, borne_inf, borne_sup, noeud_en_bas, -444, poids_en_bas, poids_en_haut, poids_arbre)





def trouve_les_aretes_cuttables(parent, enfants, liste_adj, part_en_c, c1, c2, Lc, borne_inf, borne_sup, racine, poids_de_l_arbre):
    
    

    tab_des_poids = calcule_tous_les_poids_de_maniere_utile(parent, enfants, Lc, racine, [-666] * len(parent))
    print("poids de l'arbre et de la racine")
    print(poids_de_l_arbre)
    print(tab_des_poids[racine])
    res = []

    #il faut avoir un tableau de taille taille de l'arbre tq en position i il y a le poids du sous arbre enraciné en i....


    # #sinon on peut prendre un enfant qcq de la racine, au choix.....
    # debut = random.randint(0, len(parent) - 1)
    # while parent[debut] == -1 or parent[debut] == -2 : 
    #     debut = random.randint(0, len(parent) -1)
    # #donc on a trouvé notre racine dans l'arbre, et ça c'est cool ! 

    #finalement j'ai changé d'avis : ça doit être mieux de prendre une feuille de l'arbre comme ça on ne parcourt pas tout l'arbre pour calculer son poids initial
    debut = enfants.index([]) #c'est sacrément déterministe cet histoire, mais on s'en fiche complètement donc bon

    return fonction_recursive_premiere_etape(parent, enfants, Lc, borne_inf, borne_sup, debut, -444, popu(Lc[debut]), poids_de_l_arbre - popu(Lc[debut]), tab_des_poids, True)




#liste adj : liste adjecance du graphe, part en c : tableau de taille nc qui assigne chaque canton a sa circo en cours
#retourne une part_en_c à jour 
#question de multigraphe pour choisir les deux circos à fusionner
def un_tour_de_Recom(liste_adj, part_en_c, moyenne_en_c, Lc, epsilon, nb_c):
    borne_inf = moyenne_en_c* (1-epsilon)
    borne_sup = moyenne_en_c * (1+epsilon)


    # print("on fait un tour de Recom")
    #on choisit les deux circos à fusionner en prenant une arête dans la coupe
    nb_cantons = len(liste_adj)
    c1 = random.randint(0,nb_cantons-1)
    c2 = random.randint(0,nb_cantons-1)

    #graphe non orienté
    # print("partition_en_cours")
    # print(part_en_c)
    while (part_en_c[c1] == part_en_c[c2] or c1 not in liste_adj[c2]):
        c1 = random.randint(0,nb_cantons-1)    
        c2 = random.randint(0,nb_cantons-1)   

    # ci est le numéro inscrit dans part_en_c des circos à fusionner 
    c1 = part_en_c[c1]
    c2 = part_en_c[c2]   

    ancien_T1, ancien_T2 = fonction_simple(part_en_c, c1, c2)
    # ancien_poid_1 = taille_arbre(ancien_T1, Lc)
    # ancien_poid_2 = taille_arbre(ancien_T2, Lc)

    
    # print("les circos à fusionner")
    # print(c1)
    # print(c2)  

    #on a nos deux circos à merge
    Cuttable = False
    

    #la cut_edge sera unique -< non en fait
    cut_edge = []
    # print("on cherche un truc cuttable")
    # print(part_en_c)
    # aretes_qui_coupent_nouvel_algo = []
    while not Cuttable:
        # if Cuttable == False and aretes_qui_coupent_nouvel_algo != [] and aretes_qui_coupent_nouvel_algo != None:
        #     print(T)
        #     print("aretes qui coupent nouvel algo")
        #     print(aretes_qui_coupent_nouvel_algo)
        #     raise ZeroDivisionError



        # print("on cherche le truc cuttable en générant des arbres aléas")
    # for _ in range(2):
        T, T_enfants, poids_de_l_arbre, racine = algo_de_wilson(liste_adj, part_en_c, c1, c2, Lc)
        

        # T = [16, -2, 15, 17, 7, 15, -2, 14, -2, -2, 2, -2, -2, 4, 15, -1, 3, 7, -2]
        # T_enfants = [[], [], [10], [16], [13], [], [], [17, 4], [], [], [], [], [], [], [7], [14, 2, 5], [0], [3], []]
        # poids_de_l_arbre = 209144
        # racine = 15
        # print("genere un arbre alea avec kruskal !")
        # T = kruskal_poids_alea(liste_adj, part_en_c, c1, c2)
        # print(T)


        # aretes_qui_coupent_nouvel_algo = trouve_les_aretes_cuttables(T, T_enfants, liste_adj, part_en_c, c1, c2, Lc, borne_inf, borne_sup, racine, poids_de_l_arbre)
        # print(T)
        # print(T_enfants)
        # print(poids_de_l_arbre)
        # print(racine)
        for i in range(len(T)): 
            

            #on peut supposer que Ti seront des tableaux de bool pour dire si dedans ou pas
            T1, T2 = partition_en_deux_arbres(liste_adj, part_en_c, c1, c2, i, T)
            # print("voivi les tailles respectives de t1, t2 et T")
            # print(T1)
            # print(T2)
            taille1 = taille_arbre(T1, Lc)
            taille2 = taille_arbre(T2, Lc)
            
            # if aretes_qui_coupent_nouvel_algo != None and i in aretes_qui_coupent_nouvel_algo:
            #     print("les tailllllles")
            #     print(taille1)
            #     print(taille2)


            # print(taille1)
            # print(taille2)
            # print("taille de T")
            T_prime = [False] *len(T1)
            for i in range (len(T1)):
                if T1[i]:
                    T_prime[i] = True
                if T2[i]:
                    T_prime[i] = True
            # print(T_prime)
            # print(taille_arbre(T_prime, Lc))

            moy_prime = moyenne_en_c
            
            # print((1-epsilon) * moy_prime)
            # print((1+epsilon) * moy_prime)
            # print(taille1)
            # print(taille2)
            if (1-epsilon) * moy_prime <= taille1 <= (1+epsilon) * moy_prime and (1-epsilon) * moy_prime <= taille2 <= (1+epsilon) * moy_prime :
            
            #if taille_arbre(T1, Lc) - taille_arbre(T2, Lc) < epsilon * taille_arbre(T, Lc):
                cut_edge.append((i, T1, T2, moy_prime))
                Cuttable = True
    # print("on a trouvé un truc cutable, youpi")
    ind_random = random.randint(0, len(cut_edge) -1)


    # print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
    # print(T)
    # print(cut_edge)
    # print(aretes_qui_coupent_nouvel_algo)

    cut, T1, T2, moy_en_c = cut_edge[ind_random]
    for i in range(len(T1)):
        if T1[i] :
            part_en_c[i] = c1
    for i in range(len(T2)):
        if T2[i] :
            part_en_c[i] = c2

    return part_en_c, moy_en_c

def moyenne_initiale(decoupage_inital, nb_circ, Lc):
    # for i in range(nb_circ):
    #     somme = 0
    #     for j in range(len(decoupage_inital)):
    #         if decoupage_inital[j] == i:
    #             somme += popu(Lc[i])
    #     somme = 
    somme = 0
    for j in range(len(decoupage_inital)):
        somme += popu(Lc[j])
    print("somme")
    print(somme)
    return somme/nb_circ



def genere_un_decoupage_selon_la_decomp_recom(liste_adj, Lc, epsilon, decoupage_initial, nb_tours, nb_circ): 
    # Définition de la partition initiale: 

    #v1 : on récupère le premier dans le fichier créé par exhaustifmoinsbourrin xD
    # donc il est donnée en paramètre
    decoupage = decoupage_initial
    #on calcule la première moy_en_v : 
    print("découpage en entrée de recom : ")
    print(decoupage)
    
    moy_en_c = moyenne_initiale(decoupage_initial, nb_circ, Lc)
    print(moy_en_c)

    for i in range (nb_tours):
    #for i in range(5):
        # print("après un tour de recom on a trouvé : ")
        # print(decoupage)
        decoupage, moy_en_c = un_tour_de_Recom(liste_adj, decoupage,moy_en_c, Lc, epsilon, nb_circ)

    print("découpage renvoyééééé")
    print(decoupage)
    for i in range(len(decoupage)):
        decoupage[i] = int(decoupage[i])
    return decoupage



#pas de moi : 
#liste, nbCantons, listeCantons, nbSegments, listeSegments, nbCirco

def calculsortie(L,nbc,Lc,nbs,Ls,nbcirc):
    monFichier=open(nomFichierSortie,"w")
    poputotale=0
    for i in range(nbc):
        poputotale=poputotale+popu(Lc[i])
    if int(numDep) < 10 : 
        monFichier.write("0" + numDep + "\n")# + str(nbcirc) + " ")   
    else :
        monFichier.write(numDep + "\n")# + str(nbcirc) + " ")    
    monFichier.write(str(len(L))+"\n") #combien de decoupages


    #Guti : normalement ici il n'y aura qu'un elt dans L, puisqu'on ne génère qu'un découpage à chaque fois
    for li in L:
        #monFichier.write("Sol"+"\t"+"FracPop"+"\t")
        ## Calcul de la population totale par circonscription
        popula=[0]*nbcirc

        #on écrit le découpage en question
        for ci in li:
            monFichier.write(str(ci)+" ")


        for j in range(nbc):
            popula[li[j]]=popula[li[j]]+popu(Lc[j])
        frac=[0]*nbcirc
        for j in range(nbcirc):
            frac[j]=nbcirc*1.0*popula[j]/poputotale
            monFichier.write(str(round(frac[j],3))+" ")
        
        
        monFichier.write("\n")
        
        ## Calcul du perimetre de chaque circonscription
        #perim=perimetreDecoupage(li,nbcirc,nbs,Ls)
        #perimtotal=0
        #for j in range(nbcirc):
        #    perimtotal=perimtotal+perim[j]
        #monFichier.write("PerimTot"+"\t"+str(perimtotal)+"\t")
        #monFichier.write("Num circonscription par canton \t")
    monFichier.close()



def indice_decoupage(tab_decoup, decoup, nb_circ):
    for i in range(nb_circ):
        for j in range(len(tab_decoup)):
            if tab_decoup[j] == decoup:
                return j 
        for elt in tab_decoup:
            for j in range(len(elt)):
                elt[j] = (elt[j] + 1 ) % nb_circ

    raise ZeroDivisionError


def main():
    
    
    print("on se lance dans le main ")

    # à changer si on n'est pas sur les mêmes fichiers .out !!
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    print("combien de tours :   ")

    #on fixe le nombre de tours à la main ici 1000
    nb_tours = 1000 # int(input())
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)
    epsilon = taux
    print("decoupage initial")
    print(decoup_init[0])
    L = genere_un_decoupage_selon_la_decomp_recom(liste_adj, Lc, epsilon, decoup_init[0], nb_tours, nci )
    calculsortie([L], nc, Lc, ns, Ls, nci)
    print("a priori tout à été écrit dans le fichier : " + nomFichierSortie)
    exit(0)


def formalise_decoupage( decoupage, nb_circo):
    dico = {}
    nb_en_c = 0
    ind_en_c= 0
    while len(dico) < nb_circo and ind_en_c < len(decoupage):
        if decoupage[ind_en_c] not in dico:
            dico[decoupage[ind_en_c]] = nb_en_c
            nb_en_c += 1
        ind_en_c += 1

    print(dico)


    #donc on a créé un dico de remplacement
    res = []
    for elt in decoupage:
        res.append(dico[elt])
    return res
    



def calcule_inegalite(nom_fichier_tableau_sample):

    nombre_total_decoupage = 100



    fichier = open(nom_fichier_tableau_sample, "r")
    lignes = fichier.readlines()
    nb_decoupages = int(lignes[1])
    print("\n\n")

    print("nb decoupages possibles : " + str(nb_decoupages))
    print("nb samples : (à changer manuellement) " + str(nombre_total_decoupage))
    print("nb iterations : 1000 ( à changer manuellement)")
    print("nb_dpt : " + lignes[0])
    for i in range(10):
        ligne1 = random.randint(0, nb_decoupages-1)
        ligne2 = random.randint(0, nb_decoupages-1)
        if int(lignes[ligne2+2].split()[-2]) != 0 :
            partie_droite = int(lignes[ligne1+2].split()[-2]) / int(lignes[ligne2+2].split()[-2])
        else :
            partie_droite = "infinity (proba denomi nulle)"
        
        taille_coupe_1 = int(lignes[ligne1+2].split()[-1])
        if taille_coupe_1 < 0:
            taille_coupe_1 = -1 * taille_coupe_1

        taille_coupe_2 = int(lignes[ligne2+2].split()[-1])
        if taille_coupe_2 < 0:
            taille_coupe_2 = -1 * taille_coupe_2

        partie_gauche = 2**(taille_coupe_2/taille_coupe_1)

        print("n° " + str(ligne1) + " avec coupe de taille : " + str(taille_coupe_1) + " n° " + str(ligne2) + " avec coupe de taille : " + str(taille_coupe_2) + " partie gauche : (frac) : " + str(partie_gauche) + "  " + str(partie_droite)+ " partie droite : (exp) ")

# calcule_inegalite("trucs_interressants/dpt_9_100_samples_1000_iterations_depuis_config_12.txt")
# calcule_inegalite("trucs_interressants/dpt_50_100_samples_1000_iterations_depuis_config_34.txt")

def genere_distribution(nb_decoupages, indice_debut): #, nomFichierEntree, nomFichierDecoupInit, nomFichierSortie):



    print("on se lance dans le genere_distribution ")

    # à changer si on n'est pas sur les mêmes fichiers .out !!
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    #ces deux lignes ne servent à rien, sont overlap par le paramètre !! en fait ce sont deux trucs différents
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    nb_tours = 10 #int(input())
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)
    print(mat_adj)
    print(liste_adj)
    epsilon = taux
    print("decoupage initial")
    print(decoup_init[indice_debut])
    raise ZeroDivisionError

    monFichier=open(nomFichierSortie,"w")
    F = []
    for i in range(nb_decoupages):
        print("~~~~~~~~~~~~~~~~~~~~~~~~ " + str(i) + " ~~~~~~~~~~~~~~~~~~~~~~")
        F = (genere_un_decoupage_selon_la_decomp_recom(liste_adj, Lc, epsilon, decoup_init[indice_debut], nb_tours, nci))
        F_prime = formalise_decoupage(F, nci)
        card_coupe = calcule_card_de_la_coupe(F_prime, mat_adj)

        


        for li in F_prime:
            #monFichier.write("Sol"+"\t"+"FracPop"+"\t")
            ## Calcul de la population totale par circonscription
            # popula=[0]*nci
            #on écrit le découpage en question




            monFichier.write(str(li)+" ") 
        print(card_coupe)
        monFichier.write(str(card_coupe))
        monFichier.write("\n")

    monFichier.write("decoup_initiale :  " + str(formalise_decoupage(decoup_init[indice_debut], nci)))

    # L = [0] * len(decoup_init)

    # for i in range(nb_decoupages):
    #     print("on en a généré "+ str(i) + "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    #     sample = (genere_un_decoupage_selon_la_decomp_recom(liste_adj, Lc, epsilon, decoup_init[indice_debut], nb_tours, nci ))
    #     print(decoup_init)
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     # print(sample)
    #     # for decalage in range(nci):
    #     indice_decoup = indice_decoupage(decoup_init, sample, nci)
    #     print(sample)
    #         # if sample in decoup_init:
    #         #     print("decalage : " + str(decalage))
    #         #     print(sample)
    #         #     indice_decoup = decoup_init.index(sample)
    #         #     break
    #         # else :
    #         #     for ind in range(len(sample)):
    #         #         sample[ind] = (sample[ind] + 1) % nci
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~é")
    #     print(indice_decoup)
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     assert(0<= indice_decoup < len(L))
    #     L[indice_decoup] += 1


    # monFichier=open(nomFichierSortie,"w")
    # poputotale=0
    # monFichier.write(str(nb_decoupages) + " " + str(indice_debut) + "distribution avec " + "x échantillons en partant du découpage y \n")
    # for i in range(nc):
    #     poputotale=poputotale+popu(Lc[i])
    # if int(numDep) < 10 : 
    #     monFichier.write("0" + numDep + "\n")# + str(nbcirc) + " ")   
    # else :
    #     monFichier.write(numDep + "\n")# + str(nbcirc) + " ")    
    # monFichier.write(str(len(L))+"\n\n") #combien de decoupages possibles officiels


    # #Guti : normalement ici il n'y aura qu'un elt dans L, puisqu'on ne génère qu'un découpage à chaque fois

    # for li in L:
    #     #monFichier.write("Sol"+"\t"+"FracPop"+"\t")
    #     ## Calcul de la population totale par circonscription
    #     # popula=[0]*nci
    #     monFichier.write(str(li))
    #     #on écrit le découpage en question
    #     # for ci in li:
    #     #     monFichier.write(str(ci)+" ")


    #     # for j in range(nc):
    #     #     popula[li[j]]=popula[li[j]]+popu(Lc[j])
    #     # frac=[0]*nci
    #     # for j in range(nci):
    #     #     frac[j]=nci*1.0*popula[j]/poputotale
    #     #     monFichier.write(str(round(frac[j],3))+" ")
        
        
    #     monFichier.write("\n")
        
        ## Calcul du perimetre de chaque circonscription
        #perim=perimetreDecoupage(li,nbcirc,nbs,Ls)
        #perimtotal=0
        #for j in range(nbcirc):
        #    perimtotal=perimtotal+perim[j]
        #monFichier.write("PerimTot"+"\t"+str(perimtotal)+"\t")
        #monFichier.write("Num circonscription par canton \t")
    monFichier.close()
    print(nb_tours)
    
    print("a priori tout à été écrit dans le fichier : " + nomFichierSortie)
    
def fait_le_tableau():

    #pour avoir la mat d'adj
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)
    epsilon = taux




    mon_fichier = open(nomFichierSortie, "r")
    fichier_ecrire = "avec_dist_recom/tableau_sortie_Recom" + str(numDep) + ".txt"
    fichier_source = open(nomFichierDecoupInit, "r")
    fichier_sortie = open(fichier_ecrire, "w")
    decoupages_possibles_avant = fichier_source.readlines()
    a_supprimer = int(decoupages_possibles_avant[1].split()[0])
    decoupages_possibles = decoupages_possibles_avant[2:]
    fichier_source.close()

    lignes = mon_fichier.readlines()
    mon_fichier.close()
    dico = {}
    
    for elt in decoupages_possibles : 
        #si les decoupages ne sont pas encore normalisés : 
        
        e = elt.split()[a_supprimer:]
        # print(e) 
        # raise ZeroDivisionError
        
        e = e
        rr = formalise_decoupage(e, nci)
        taille_coupe = calcule_card_de_la_coupe(rr, mat_adj)

        CC = []
        for j in range(a_supprimer):
            CC.append([])
        for j in range(len(rr)):
            # print(int(decoupage[j]))
            CC[int(rr[j])].append(j)

        nombre_forets_couvrantes = round(calcule_nombre_de_foret_couvrante(mat_adj, CC))


        print(rr)
        elt = ""
        for r in rr:
            elt = elt + str(r) + " "

        elt = elt[:-1] #voir s'il faut mettre -1, -2 ou -3, dépend du format du fichier

        #on considère que les découpages sont déja formalisés
        print("ééééé" + str(elt) + "ééééé")
        assert(elt not in dico)
        dico[elt] = (0,nombre_forets_couvrantes, taille_coupe)
        

    print("dicoo")
    print(dico)
    print(lignes)
    for elt in lignes[:-1] :
        taille_coupe = elt.split()[-1]
        
        tmp = elt.split()[:-1]
        elt = ""
        for f in tmp:
            elt = elt + f + " "
        elt = elt[:-1]



        # elt = elt[:-3] # voir s'il faut mettre -1, -2 ou -3, depend du format du fichier
        print("eltttt")
        print("eeee" + elt + "eeeeeee") 
        assert(elt in dico)
        dico[elt] = (1+dico[elt][0], dico[elt][1], taille_coupe)

    fichier_sortie.write(str(numDep) + "\n")
    fichier_sortie.write(str(len(dico)) + "\n")


    for elt in dico:

        fichier_sortie.write(elt + " ")
        #le nomre d'occurences
        print(dico[elt][0])
        fichier_sortie.write(str(round(dico[elt][0])) + " ")
        #le nombre de spanning tree
        print(dico[elt][1])
        fichier_sortie.write(str(round(dico[elt][1])) + " ")
        #le cardinal de la coupe
        print(dico[elt][2])
        fichier_sortie.write(str(dico[elt][2]) + "\n")

    fichier_sortie.write(lignes[-1])

    


    fichier_sortie.close()

def calcule_card_de_la_coupe(decoupage, mat_adj):
    res = 0
    for i in range(len(mat_adj)):
        for j in range(i+1, len(mat_adj)): #les boucles n'existent pas
            if mat_adj[i][j] == 1 and decoupage[i] != decoupage[j]: 
                res += 1
    return res

def converti_en_tableau_de_tableaux(dico, nb_elts_a_suppr):

    
    res = [] #contient les valeurs à mettre dans l'histo
    res_bis = [] #contient les découpages dans le même ordre
    tab_taille_coupe = [] #contient les tailles des coupes à mettre en abscisse
    clefs = sorted(dico.keys())[:len(dico.keys())-nb_elts_a_suppr]
    petit_tableau = [0]*len(clefs)
    petit_tableau_bis = [""] * len(clefs)
    for j in range(len(clefs)):
        elt = clefs[j]
        tab_taille_coupe.append(elt)
        decoup_de_coupe_fixee_et_occurrence_associee = dico[elt]
        if len(decoup_de_coupe_fixee_et_occurrence_associee) > len(res):
            while (len(decoup_de_coupe_fixee_et_occurrence_associee)) > len(res):
                res.append([0]*len(clefs))
                res_bis.append([""]*len(clefs))
        for i in range(len(decoup_de_coupe_fixee_et_occurrence_associee)):
            res[i][j] = decoup_de_coupe_fixee_et_occurrence_associee[i][1]
            res_bis[i][j] = decoup_de_coupe_fixee_et_occurrence_associee[i][0]

    return res, res_bis, tab_taille_coupe



def fait_histogramme():
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)
    epsilon = taux




    fichier_ecrire = "avec_dist_recom/tableau_sortie_Recom" + str(numDep) + ".txt"
    fichier_tableau = open(fichier_ecrire, "r")
    decoupages_tableau = fichier_tableau.readlines()
    # a_supprimer = int(decoupages_possibles_avant[1].split()[0])
    # decoupages_possibles = decoupages_possibles_avant[2:] 

    #changement de convention ??
    nb_decoupages = int(decoupages_tableau[1].split()[0])
    fichier_tableau.close()

    
    dico = {}

    for i in range(2, nb_decoupages+2):
        decoup_en_c = decoupages_tableau[i].split()
        taille_coupe = int(decoup_en_c[-1])
        if taille_coupe < 0:
            taille_coupe = (-1) * taille_coupe
        if taille_coupe in dico : 

            #format : découpage, nombre d'apparition du découpage
            dico[taille_coupe].append((decoup_en_c[:-2], (int(decoup_en_c[-2]))))
        else :
            dico[taille_coupe] = [(decoup_en_c[:-2], int(decoup_en_c[-2]))]



    print(dico)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    tab_ordonne, t_decoupages_associes, tab_taille_coupe = converti_en_tableau_de_tableaux(dico, 0)
        


    #bidouille : 
    pour_histo =[]
    for i in range(len(tab_ordonne)):
        pour_histo.append([])
        for j in range(len(tab_ordonne[i])):
            elt_bis= tab_ordonne[i][j]
            for cpt in range(elt_bis):
                pour_histo[i].append(tab_taille_coupe[j])


    
    print(tab_ordonne)
    #print(t_decoupages_associes)
    print(tab_taille_coupe)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs = np.random.randint(0, 10, 50)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs2 = np.random.randint(0, 10, 50)
    # # Création d'un tableau avec les intervalles centrés sur la valeur entière
    # inter = np.linspace(-0.5, 9.5, 11)

    plt.hist(pour_histo, bins=tab_taille_coupe, rwidth=0.8, stacked=True,
            label=['Valeurs 1', 'Valeurs 2'])  # Création de l'histogramme
    plt.xlabel('Taille de la coupe (regarder le nb à gauche)')
    plt.xticks(tab_taille_coupe)
    plt.ylabel('Nombre de découpage (une couleur = un découpage)')
    plt.title("Dpt : " + str(numDep) + " , 5%, Nombre de découpage en fct de la coupe")
    plt.legend()
    
    plt.savefig("graphiques/nb_occ_en_fct_taille_coupe/" + str(numDep) + ".png")
    # plt.show()



#fait l'histogramme nombre d'occurence en fonction du nombre de spanning forest
def fait_histogramme_bis():
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)
    epsilon = taux




    fichier_ecrire = "avec_dist_recom/tableau_sortie_Recom" + str(numDep) + ".txt"
    fichier_tableau = open(fichier_ecrire, "r")
    decoupages_tableau = fichier_tableau.readlines()
    # a_supprimer = int(decoupages_possibles_avant[1].split()[0])
    # decoupages_possibles = decoupages_possibles_avant[2:] 
    nb_decoupages = int(decoupages_tableau[1])
    fichier_tableau.close()

    
    dico = {}

    for i in range(2, nb_decoupages+2):
        decoup_en_c = decoupages_tableau[i].split()
        nb_f = int(decoup_en_c[-2])
        
        if nb_f in dico : 

            #format : découpage, nombre d'apparition du découpage
            dico[nb_f].append((decoup_en_c[:-3], int(decoup_en_c[-3])))
        else :
            dico[nb_f] = [(decoup_en_c[:-3], int(decoup_en_c[-3]))]



    print(dico)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    tab_ordonne, t_decoupages_associes, tab_taille_coupe = converti_en_tableau_de_tableaux(dico, 0)
        


    #bidouille : 
    pour_histo =[]
    for i in range(len(tab_ordonne)):
        pour_histo.append([])
        for j in range(len(tab_ordonne[i])):
            elt_bis= tab_ordonne[i][j]
            for cpt in range(elt_bis):
                pour_histo[i].append(tab_taille_coupe[j])


    
    print(tab_ordonne)
    #print(t_decoupages_associes)
    print(tab_taille_coupe)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs = np.random.randint(0, 10, 50)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs2 = np.random.randint(0, 10, 50)
    # # Création d'un tableau avec les intervalles centrés sur la valeur entière
    # inter = np.linspace(-0.5, 9.5, 11)

    plt.hist(pour_histo, bins=tab_taille_coupe, rwidth=0.8, stacked=True,
            label=['Valeurs 1', 'Valeurs 2'])  # Création de l'histogramme
    plt.xlabel('Nombre spanning tree (regarder le nb à gauche)')
    plt.xticks(tab_taille_coupe)
    plt.ylabel('Nombre de découpage tirés (une couleur = un découpage)')
    plt.title("Dpt : " + str(numDep) + " , 5%, Nombre de découpage en fct de la coupe")
    plt.legend()
    plt.show()
    plt.savefig("graphiques/nb_occ_en_fct_nb_spanning_forest/" + str(numDep) + ".png")
#

def algo_matrix_tree_theorem(mat_adj, liste_adj):
    lap_mat = mat_adj
    # lap_mat = []
    # for i in range(len(mat_adj)):

    #     degre = 0
    #     for j in range(len(mat_adj)):
    #         if mat_adj[i][j] != 0:
    #             degre += 1

    #     lap_mat.append([0] * len(mat_adj))
    #     for j in range(len(mat_adj)):
    #         if i == j : 
    #             # lap_mat[i][i] = len(liste_adj[i])
    #             lap_mat[i][i] = degre
    #         elif mat_adj[i][j] == 1:
    #             lap_mat[i][j] = -1

    lap_mat = np.array(lap_mat)

    # ligne_col_supp = random.randint(len(lap_mat) -1 )
    ligne_col_supp = 0

    lap_mat = np.delete(lap_mat, ligne_col_supp, 0)
    lap_mat = np.delete(lap_mat, ligne_col_supp, 1)

    nb_spanning_tree = np.linalg.det(lap_mat)
    return nb_spanning_tree

def visite_prof(mat_adj, noeud, liste_retour, liste_visite):
    liste_visite[noeud] = True

    for i in range(len(mat_adj)):

        if mat_adj[noeud][i] == 1 and not(liste_visite[i]):
            liste_retour, liste_visite = visite_prof(mat_adj, i, liste_retour, liste_visite)
    liste_visite.append(noeud)
    return (liste_retour, liste_visite)

def parcours_prof(mat_adj):
    liste_visite = [False] * len(mat_adj)
    liste_retour = []

    for i in range(len(mat_adj)):
        if not(liste_visite[i]):
            liste_retour, e = visite_prof(mat_adj, i, liste_retour, [])
            liste_visite.append(e)
    return liste_visite

# def visite_prof_ret(mat_adj, noeud, liste_retour, liste_visite):
#     liste_visite[noeud] = True

#     for i in range(len(mat_adj)):

#         if mat_adj[i][noeud] == 1 and not(liste_visite[i]):
#             liste_retour, liste_visite = visite_prof(mat_adj, i, liste_retour, liste_visite)
#     liste_visite.append(noeud)
#     return (liste_retour, liste_visite)

# def parcours_prof_ret(mat_adj, ordre):
#     liste_visite = [False] * len(mat_adj)
#     liste_retour = []

#     for i in ordre:
#         if not(liste_visite[i]):
#             liste_retour, liste_visite = visite_prof(mat_adj, i, liste_retour, liste_visite)

#     return liste_visite


# def kosaraju(mat_adj):



def calcule_nombre_de_foret_couvrante(mat_adj, CC):
    # composantes_connexe =parcours_prof(mat_adj)
    composantes_connexe = CC

    res = 1

    for elt in composantes_connexe:
        mat_adj_bis = []
        for j in range(len(elt)):
            mat_adj_bis.append([0] * len(elt))

        for elt_un in range(len(elt)):
            for elt_deux in range(len(elt)):
                if elt_un != elt_deux and mat_adj[elt[elt_un]][elt[elt_deux]] == 1:
                    mat_adj_bis[elt_un][elt_deux] = -1 
        for elt_un in range(len(elt)) :
            degre = 0
            for j in range(len(elt)):
                if mat_adj_bis[elt_un][j] != 0:
                    degre += 1
            mat_adj_bis[elt_un][elt_un] = degre

        # print(mat_adj_bis)

        nb_arbre_couv = round(algo_matrix_tree_theorem(mat_adj_bis, []))
        # print("nb arbre couv")
        # print(nb_arbre_couv)
        res *= nb_arbre_couv
    return res


def calcule_une_fois_pour_toute_le_nombre_de_spanning_tree(numDepb = None):
    # ~~ pour automatiser sur tous les découpages :
    if numDepb != None : 
        numDep = int(numDepb) + 1

    if numDep == 20 or numDep == 69 or numDep == 75 : 
        numDep += 1

    nomFichierEntree = "graphe_depts/"+ str(numDep) + ".out"


    if numDep > 9 :
        nomFichierDecoupInit = "actuel-5pour100/sortie"+ str(numDep) + ".txt"
    else : 
        nomFichierDecoupInit = "actuel-5pour100/sortie0"+ str(numDep) + ".txt"


    print(nomFichierEntree)
    print(nomFichierDecoupInit)
    print(numDep)

    # ~~ fin de l'automatisation


    #on lit dans nom_fichier_entree
    #on écrit dans graphe_depts/n°_bis.out

    nc, Lc, ns, Ls, decoup_init = lecture_bis(nomFichierEntree, nomFichierDecoupInit)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(nc)
    print(Lc)
    print(ns)
    print("~~~~~~~~~~~~~~~~~~~~~~~é")

    print("combien de tours :   ")
    #nb_tours = int(input())
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)

    mon_fichier = open(nomFichierDecoupInit, "r")
    nomFichierSortie = "actuel-5pour100/sortie" + str(numDep) + "_bis.txt"
    fichier_sortie = open(nomFichierSortie,"w")

    lignes = mon_fichier.readlines()
    mon_fichier.close()

    fichier_sortie.write(lignes[0])

    #NB : elt_a_suppr est égal au nombre de circonscriptions, car en fébut de ligne on met les écarts à la moyenne en pop de chaque circo
    elt_a_suppr = int(lignes[1].split()[0])

    # ~~ pour automatisation

    nci = elt_a_suppr
    # ~~ fin automatisation

    nb_dec = int(lignes[1].split()[1])


    fichier_sortie.write(str(elt_a_suppr + 2) + " ")
    fichier_sortie.write(str(nb_dec) + "\n")

    for i in range(nb_dec):
        elts_a_suppr = lignes[i+2].split()[:elt_a_suppr]
        decoupage = lignes[i+2].split()[elt_a_suppr:]

       
        CC = []
        for j in range(elt_a_suppr):
            CC.append([])

        vrai_dec = []
        for j in range(len(decoupage)):
            # print(int(decoupage[j]))
            CC[int(decoupage[j])].append(j)
            vrai_dec.append(int(decoupage[j]))
        print(decoupage)
        print(mat_adj)
        card_coupe = calcule_card_de_la_coupe(decoupage, mat_adj)

        print(CC)
        nb_spanning_forest = calcule_nombre_de_foret_couvrante(mat_adj, CC)

        for elt in elts_a_suppr:
            fichier_sortie.write(elt + " ")
        print("nombre de forets couvrantes")
        print(nb_spanning_forest)

        fichier_sortie.write(str(card_coupe) + " ")
        fichier_sortie.write(str(nb_spanning_forest) )

        for elt in decoupage:
            fichier_sortie.write(" " + elt)
        fichier_sortie.write("\n")






    fichier_sortie.close()
    return numDep



def represente_taille_coupe_par_rapport_a_nb_spanning_tree(numDepb):

    if numDepb != None : 
        numDep = int(numDepb) + 1

    if numDep == 20 or numDep == 69 or numDep == 75 : 
        numDep += 1



    nomFichierEntree = "actuel-5pour100/sortie" + str(numDep) + "_bis.txt"
    mon_fichier = open(nomFichierEntree, "r")

    lignes = mon_fichier.readlines()
    mon_fichier.close()

    a_supp = int(lignes[1].split()[0])
    nb_dec = int(lignes[1].split()[1])

    tailles_coupes = []
    nb_spanning_forests = []
    nombre_occurence = []
    dico = {}
    for i in range(nb_dec):
        decoupage = lignes[i+2].split()[a_supp:]
        taille_coupe = int(lignes[i+2].split()[a_supp -2])
        tailles_coupes.append(taille_coupe)
        
        # nombre_occurence.append(taille_coupe)
        nb_spanning_forest = int(lignes[i+2].split()[a_supp-1])
        
        if taille_coupe in dico:
            dico[taille_coupe] += 1
        else :
            dico[taille_coupe] = 1
        nombre_occurence.append(dico[taille_coupe])
        nb_spanning_forests.append(np.log(nb_spanning_forest))



    #V1 avec seulement un y-axis et sans légendes
    # plt.plot(tailles_coupes,nb_spanning_forests,"ob") # ob = type de points "o" ronds, "b" bleus


    #v2 avec deux y-axis et avec légende
    fig, ax1 = plt.subplots()
    ax1.plot(tailles_coupes, nb_spanning_forests, "xg", label = "ax1 plot")
    ax1.set_xlabel("cardinal de la coupe")
    ax1.set_ylabel("nombre de foret couvrante", color = 'b')

    ax2 = ax1.twinx()
    # nombre_occurence = []
    # for elt in sorted(dico.keys()):
    #     nombre_occurence.append(dico[elt])
    ax2.plot(tailles_coupes, nombre_occurence, ".r", label = "ax2 plot")
    ax2.set_ylabel("nombre de découpages", color = 'b')

    fig.suptitle("Département " + str(numDep) + ", 5%" )
    plt.savefig("graphiques/nb_spanning_tree_en_fonction_taille_coupe/dpt_" + str(numDep) + ".png")
    # plt.show()
    return numDepb

def represente_nombre_decoupage_en_fonction_nombre_de_spanning_tree(nb_elts__de_la_fin_a_suppr):
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    mat_adj = graphadj(nc, Lc, ns, Ls)
    liste_adj = listeadj(mat_adj)
    epsilon = taux



    fichier_avec_distribution = "avec_dist_recom/tableau_sortie_Recom" + numDep + ".txt"
    fichier_tableau = open(fichier_avec_distribution, "r")
    decoupages_tableau = fichier_tableau.readlines()
    # a_supprimer = int(decoupages_possibles_avant[1].split()[0])
    # decoupages_possibles = decoupages_possibles_avant[2:] 
    nb_decoupages = int(decoupages_tableau[1])
    fichier_tableau.close()

    
    dico = {}
    liste_nb_spanning_tree = []
    ordonnee_nb_spanning_tree = []

    for i in range(2, nb_decoupages+2):
        decoup_en_c = decoupages_tableau[i].split()[:-2]
        dec = []
        for elt in decoup_en_c:
            dec.append(int(elt))

        CC = []
        for i in range(nci):
            CC.append([])
        for i in range(len(decoup_en_c)):
            CC[int(decoup_en_c[i])].append(i)
        nb_spanning_trees = calcule_nombre_de_foret_couvrante(mat_adj, CC)
        liste_nb_spanning_tree.append(nb_spanning_trees)
        ordonnee_nb_spanning_tree.append(int(decoup_en_c[-2]))



        taille_coupe = int(decoup_en_c[-1])
        if taille_coupe < 0:
            taille_coupe = (-1) * taille_coupe
        if nb_spanning_trees in dico : 

            #format : découpage, nombre d'apparition du découpage
            dico[nb_spanning_trees].append((decoup_en_c[:-2], int(decoup_en_c[-2])))
        else :
            dico[nb_spanning_trees] = [(decoup_en_c[:-2], int(decoup_en_c[-2]))]



    print(dico)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    tab_ordonne, t_decoupages_associes, tab_nb_spanning_forest = converti_en_tableau_de_tableaux(dico, nb_elts_a_suppr= nb_elts__de_la_fin_a_suppr)
        


    #bidouille : 
    pour_histo =[]
    for i in range(len(tab_ordonne)):
        pour_histo.append([])
        for j in range(len(tab_ordonne[i])):
            elt_bis= tab_ordonne[i][j]
            for cpt in range(elt_bis):
                pour_histo[i].append(tab_nb_spanning_forest[j])


    
    print(tab_ordonne)
    #print(t_decoupages_associes)
    print(tab_nb_spanning_forest)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs = np.random.randint(0, 10, 50)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs2 = np.random.randint(0, 10, 50)
    # # Création d'un tableau avec les intervalles centrés sur la valeur entière
    # inter = np.linspace(-0.5, 9.5, 11)

    plt.hist(pour_histo, bins=tab_nb_spanning_forest, rwidth=0.8, stacked=True,
            label=['Valeurs 1', 'Valeurs 2'])  # Création de l'histogramme
    plt.xlabel('Nombre de spanning forests (regarder le nb à gauche)')
    plt.xticks(tab_nb_spanning_forest)
    plt.ylabel('Nombre de découpage (une couleur = un découpage)')
    plt.title("Dpt : " + str(numDep) + " , 5%, Nombre de découpage en fct de la coupe")
    plt.legend()
    plt.show()

def calcule_inegalite_raffinee(numDep):
    k_1, k_2 = calcule_k_1_et_k2(numDep)

    fichier_avec_distribution = "avec_dist_recom/tableau_sortie_Recom" + str(numDep) + ".txt"
    mon_fichier = open(fichier_avec_distribution, "r")
    lignes = mon_fichier.readlines()
    mon_fichier.close()

    #le nombre d'occurrence est en -2, la taille de la coupe en -2

    tab_res = []
    tab_taille_coupe1 = []
    tab_taille_coupe2 = []
    tab_difference_correspondante = []
    # fait la boucle pour calculer toutes les parties gauches et droites : 
    #on ne veut pas aller voir la découp initiale qui est en dernière ligne
    for i in range(2, len(lignes)-1):
        nb_occ1 = int(lignes[i].split()[-3])
        taille_coupe1 = int(lignes[i].split()[-1])

        for j in range(2, len(lignes)-1):
            nb_occ2 = int(lignes[j].split()[-3])
            taille_coupe2 = int(lignes[j].split()[-1])
            

            if taille_coupe1 < 0 : 
                taille_coupe1 = -1 * taille_coupe1
            if taille_coupe2 < 0 : 
                taille_coupe2 = -1 * taille_coupe2

            if nb_occ2 > 0 : 
                partie_a_gauche = nb_occ1/nb_occ2
            else : 
                partie_a_gauche = "infinity"

            if taille_coupe1 == taille_coupe2 : 
                partie_a_droite = 1/(2 * k_2)
            elif k_1 == 1 : 
                partie_a_droite = "infinity"
            elif taille_coupe1 == 0 : 
                partie_a_droite = (1/ 2*k_2) * (1+1/(k_1 - 1))**(-1)

            else : 
                partie_a_droite = (1/ 2*k_2) * (1+1/(k_1 - 1))**(taille_coupe2/taille_coupe1 - 1)

            
            tab_res.append((partie_a_gauche, partie_a_droite))
            pt_droite = []
            pt_gauche = []

            if partie_a_droite != "infinity" and partie_a_gauche != "infinity" : 
                if partie_a_gauche / partie_a_droite < 3 :
                    if partie_a_gauche > 0 :  
                        print("'''''''''''''''''''''")
                        print(taille_coupe1)
                        print(taille_coupe2)
                        print(partie_a_gauche)
                        print(partie_a_droite)
                        tab_taille_coupe1.append(taille_coupe1)
                        tab_taille_coupe2.append(taille_coupe2)
                        pt_droite.append(partie_a_droite)
                        pt_gauche.append(partie_a_gauche)


                        tab_difference_correspondante.append(partie_a_gauche / partie_a_droite)
            # else : 
            #     tab_taille_coupe1.append(taille_coupe1)
            #     tab_taille_coupe2.append(taille_coupe2)
            #     tab_difference_correspondante.append(2)


    #fait le graphique
    tab_bis = [tab_taille_coupe1, tab_taille_coupe2, tab_difference_correspondante]
    # print(tab_bis)
    # print(tab_taille_coupe1)
    compil = {"Coupe1" : pt_gauche, "Coupe2" : pt_droite, "difference" : tab_difference_correspondante}
    compil = pd.DataFrame(compil)
    # print(compil)

    sns.relplot(x = "Coupe1", y = "Coupe2", hue = "difference", data = compil)
    
    plt.savefig("graphiques/difference_ineg/dpt_" + str(numDep) + ".png")
    plt.show()


    return tab_res




# def automatisation_de_calculs(liste_noms_et_nb_circos_departements):
liste_noms_et_nb_circos_departements = [(3,3), (8,3), (10,3), (11, 3), (12, 3), (16, 3), (18,3), (39,3), (40,3), (41, 3), (47,3), (53, 3), (89, 3), (61, 3), (79,3), (81,3), (87,3)]
#liste_noms_et_nb_circos_departements = [(13,16), (29,8), (31,10), (33, 12), (34,9), (35,8), (44,10), (38,10), (57,9), (59,21), (62,12), (67,9), (76,10), (77,11), (78,12), (83,8), (91,10), (92, 13), (93,12), (94,11), (95,10) ]

liste_noms_et_nb_circos_departements = [(68,6), (74,6), (38,10), (57,9), (67,9), (83,8), (94,11) ,(31,10)]
liste_noms_et_nb_circos_departements = [(87,3)]
for elt in liste_noms_et_nb_circos_departements:
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("nouveau départment")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    numDep = elt[0]
    nomFichierEntree= "graphe_depts/" + str(numDep) + ".out"

    if numDep < 10 : 

        nomFichierDecoupInit = "actuel-5pour100/sortie0" + str(numDep) + ".txt"
    else : 
        nomFichierDecoupInit = "actuel-5pour100/sortie" + str(numDep) + ".txt"

    ## Nombre de circonscriptions
    nci=elt[1]
    ## Taux d'ecart max a la moyenne (population)
    taux = 0.05

    #gestion de la Corse <>
    if(numDep == '20'):
        numDep = '2A'
        nomFichierEntree = "graphe_depts/2A.out"
    elif(numDep == '96'):
        numDep = '2B'
        nomFichierEntree = "graphe_depts/2B.out"

    #Paris et Lyon sont gérés à part 

    if(numDep == '75' or numDep == '69'):
        sys.exit(1)
    nomFichierSortie= "avec_dist_recom/sortie_Recom" + str(numDep) + ".txt"


    #arguments : nb_decoupage (ie nb de samples),indice debut
    genere_distribution(10,1)
    fait_le_tableau()
    # fait_histogramme()
    # fait_histogramme_bis()


# automatisation_de_calculs([(9,2)])

    

  
# represente_nombre_decoupage_en_fonction_nombre_de_spanning_tree(nb_elts__de_la_fin_a_suppr=5)
# calcule_une_fois_pour_toute_le_nombre_de_spanning_tree(numDep)
# represente_taille_coupe_par_rapport_a_nb_spanning_tree()


    
# numdepB = 90
# for i in range(5):
#     numdepB = calcule_une_fois_pour_toute_le_nombre_de_spanning_tree(numdepB)

# numdepB = 21
# for i in range(5):
#     numdepB = represente_taille_coupe_par_rapport_a_nb_spanning_tree(numdepB)

# represente_taille_coupe_par_rapport_a_nb_spanning_tree(11)

# represente_taille_coupe_par_rapport_a_nb_spanning_tree(16)
# represente_taille_coupe_par_rapport_a_nb_spanning_tree(17)
# represente_taille_coupe_par_rapport_a_nb_spanning_tree(20)

# represente_taille_coupe_par_rapport_a_nb_spanning_tree(94)





#arguments : nb_decoupage,indice debut
genere_distribution(10,1)
# fait_le_tableau()
# fait_histogramme()
# fait_histogramme_bis()

#fct_test()

# print("k_1_et_k_2")
# print(calcule_k_1_et_k2(2))

# print(calcule_inegalite_raffinee(5))

#intéressant : 5, 50, 4, 9, 3, 
# calcule_inegalite_raffinee(50)




    




    
    






