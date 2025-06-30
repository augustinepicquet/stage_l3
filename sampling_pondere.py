# ce fichier a pour but de faire du sampling seon recom sur des graphes pondérés : 


import matplotlib.pyplot as plt  # Module pour tracer les graphiques
import numpy as np

import math

import random



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
        if len(decoupage) != nbCantons:
            print("ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh" + str(len(decoupage)))
            #raise ZeroDivisionError
        tous_les_decoupages.append(decoup_bien_forme)
        
        

    return ([nbCantons,Lc,len(Ls),Ls, tous_les_decoupages])


def calcule_les_ponderations(numDep):
    nom_fichier = "graphe_depts/" + str(numDep) + ".out"
    fichier = open(nom_fichier, "r")
    lignes = fichier.readlines()
    fichier.close()

    dico_taille = {}
    a_suppr = int(lignes[0].split()[-1])
    for elt in lignes[a_suppr + 3 : ]:
        splitte = elt.split()
        
        f1 = int(splitte[-2]) - 1
        f2 = int(splitte[-1]) - 1
        if f1 >= 0 and f2 >= 0 : 
            var_intermediraire = np.sin(float(splitte[1])) * np.sin(float(splitte[3]))
            var_intermediraire_deux = np.cos(float(splitte[1])) * np.cos(float(splitte[3])) * np.cos(float(splitte[2]) - float(splitte[0]))
            distance = 6371 * np.arccos(var_intermediraire +  var_intermediraire_deux)

            if not (f1, f2) in dico_taille:
                dico_taille[(f1,f2)] = distance
            else : 
                dico_taille[(f1,f2)] += distance

    return dico_taille


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

## Renvoie la matrice d'adjacence du graphe des cantons
def graphadj(nc,Lc,ns,Ls, numDep):
    print("##############################################")
    print(Ls)
    dico_ponderations = calcule_les_ponderations(numDep)
    mat=[]
    print(dico_ponderations)
    for i in range(nc):
        mat.append([0]*nc)
    for seg in Ls:
        if (cant2(seg)>0): ##le canton n'est pas en limite du departement
            assert(cant1(seg) < cant2(seg))
            if (cant1(seg), cant2(seg)) in dico_ponderations:
                mat[cant1(seg)][cant2(seg)]=dico_ponderations[(cant1(seg), cant2(seg))]
                mat[cant2(seg)][cant1(seg)]=dico_ponderations[(cant1(seg), cant2(seg))]
            else : 
                mat[cant1(seg)][cant2(seg)]=dico_ponderations[(cant2(seg), cant1(seg))]
                mat[cant2(seg)][cant1(seg)]=dico_ponderations[(cant2(seg), cant1(seg))]
                
    print(mat)
    return mat

def listeadj(mat):
    n = len(mat)
    ls = [[(i, mat[j][i]) for i in range(n) if mat[j][i]>0] for j in range(n)]
    #for l in ls:
    #    print(l)
    return ls

#liste adj : liste adjecance du graphe, part en c : tableau de taille nc qui assigne chaque canton a sa circo en cours
# circ1,2 : circo à fusionner et spliter
# s : sommet dont on doit trouver un successeur random
#NB : on est dand un graphe non orienté
def random_successeur(liste_adj, part_en_c, circ1, circ2, s):
    assert(len(liste_adj) > 0)

    borne_intervalle = 0
    for elt in liste_adj[s]:
        if (part_en_c[elt[0]] == circ1 or part_en_c[elt[0]] == circ2) and elt[0] != s:
            borne_intervalle += elt[1]

    nb_alea = random.uniform(1, borne_intervalle)

    int_en_c = 0
    for elt in liste_adj[s]:
        if (part_en_c[elt[0]] == circ1 or part_en_c[elt[0]] == circ2) and elt[0] != s:
            if nb_alea <= int_en_c + elt[1]:
                return(elt[0])
            int_en_c += elt[1]
    raise ZeroDivisionError

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
                print("on a u éqal à   " + str(u))
                print("c1 et c2 : " + str(circ1) + "  " + str(circ2))
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

# T est une liste de booléen tq T[i] dit si i est dans l'arbre ou non
# Lc est le résultat de la lécture du fichier json (qui est converti en (liste/mat) d'adj par la suite)
def taille_arbre(T, Lc): 
    res = 0
    for i in range (len(T)):
        if T[i]:
            res += popu(Lc[i])
    return res

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
    liste_adj_bis = [e for (e,j) in liste_adj[c2]]
    while (part_en_c[c1] == part_en_c[c2] or c1 not in liste_adj_bis):
        c1 = random.randint(0,nb_cantons-1)    
        c2 = random.randint(0,nb_cantons-1)   

    # ci est le numéro inscrit dans part_en_c des circos à fusionner 
    c1 = part_en_c[c1]
    c2 = part_en_c[c2]   

    # ancien_T1, ancien_T2 = fonction_simple(part_en_c, c1, c2)
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

            T_prime = [False] *len(T1)
            for i in range (len(T1)):
                if T1[i]:
                    T_prime[i] = True
                if T2[i]:
                    T_prime[i] = True
            # print(T_prime)
            # print(taille_arbre(T_prime, Lc))

            moy_prime = moyenne_en_c
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

def calcule_card_de_la_coupe_pondere(decoupage, mat_adj):
    res = 0
    for i in range(len(mat_adj)):
        for j in range(i+1, len(mat_adj)): #les boucles n'existent pas
            if mat_adj[i][j] > 0 and decoupage[i] != decoupage[j]: 
                res += mat_adj[i][j]
    return res


def genere_distribution(nb_decoupages, indice_debut, nb_tours, numDep): #, nomFichierEntree, nomFichierDecoupInit, nomFichierSortie):



    print("on se lance dans le genere_distribution ")

    # à changer si on n'est pas sur les mêmes fichiers .out !!
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)
    liste_adj = listeadj(mat_adj)
    print(mat_adj)
    print(liste_adj)
    epsilon = taux
    print("decoupage initial")
    print(decoup_init[indice_debut])


    monFichier=open(nomFichierSortie,"w")
    F = []
    for i in range(nb_decoupages):
        print("~~~~~~~~~~~~~~~~~~~~~~~~ " + str(i) + " ~~~~~~~~~~~~~~~~~~~~~~")
        F = (genere_un_decoupage_selon_la_decomp_recom(liste_adj, Lc, epsilon, decoup_init[indice_debut], nb_tours, nci))
        F_prime = formalise_decoupage(F, nci)
        card_coupe = calcule_card_de_la_coupe_pondere(F_prime, mat_adj)

        


        for li in F_prime:
            monFichier.write(str(li)+" ") 
        print(card_coupe)
        monFichier.write(str(card_coupe))
        monFichier.write("\n")

    monFichier.write("decoup_initiale :  " + str(formalise_decoupage(decoup_init[indice_debut], nci)))

    # L = [0] * len(decoup_init)

    
  
    monFichier.close()
    print(nb_tours)
    
    print("a priori tout à été écrit dans le fichier : " + nomFichierSortie)


numDep = 21
nomFichierEntree = "graphe_depts/"+ str(numDep) + ".out"
nomFichierSortie = "avec_recom_pondere/sortie_Recom" + str(numDep) + ".txt"
nomFichierDecoupInit = "actuel-5pour100/sortie"+ str(numDep) + ".txt"
nci = 5
taux = 0.05


genere_distribution(2, 1, 10, numDep)

    