# ce fichier a pour but de faire du sampling seon recom sur des graphes pondérés : 
#gUVm64!fRx792

import matplotlib.pyplot as plt  # Module pour tracer les graphiques
import numpy as np

import math

import random



dico_nb_aretes_qui_coupent = [{}] 


def lecture_bis( nomFichierEntree,nomFichierDecoupInit):
    print("nomm fichierr entreee")
    print(nomFichierEntree)
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
    a_suppr = int(tableau[1][0])
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
    # print("##############################################")
    # print(Ls)
    dico_ponderations = calcule_les_ponderations(numDep)
    mat=[]
    # print(dico_ponderations)
    for i in range(nc):
        mat.append([0]*nc)
    for seg in Ls:
        if (cant2(seg)>0): ##le canton n'est pas en limite du departement
            # assert(cant1(seg) < cant2(seg))
            if (cant1(seg), cant2(seg)) in dico_ponderations:
                mat[cant1(seg)][cant2(seg)]=dico_ponderations[(cant1(seg), cant2(seg))]
                mat[cant2(seg)][cant1(seg)]=dico_ponderations[(cant1(seg), cant2(seg))]
            else : 
                mat[cant1(seg)][cant2(seg)]=dico_ponderations[(cant2(seg), cant1(seg))]
                mat[cant2(seg)][cant1(seg)]=dico_ponderations[(cant2(seg), cant1(seg))]
                
    # print(mat)
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
    # print("bruhhhhhhh")
    # print(part_en_c)
    # print(circ1)
    # print(circ2)
    # print(InTree)
    for i in range (nb_cantons):
        # print("bruhhhhhhh")
        
        if InTree[i] != -1 :
            
            u = i
            while not (InTree[u] == 0):
                # print(liste_adj)
                # print(InTree)
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
    liste_adj_bis = []
    for elt in liste_adj[c2]:
        liste_adj_bis.append(elt[0])

    while (part_en_c[c1] == part_en_c[c2] or c1 not in liste_adj_bis):
        c1 = random.randint(0,nb_cantons-1)    
        c2 = random.randint(0,nb_cantons-1)   
        liste_adj_bis = []
        for elt in liste_adj[c2]:
            liste_adj_bis.append(elt[0])

    # print(c2)
    # print(liste_adj_bis)
    # ci est le numéro inscrit dans part_en_c des circos à fusionner 
    c1 = part_en_c[c1]
    c2 = part_en_c[c2]   

    # ancien_T1, ancien_T2 = fonction_simple(part_en_c, c1, c2)
    # ancien_poid_1 = taille_arbre(ancien_T1, Lc)
    # ancien_poid_2 = taille_arbre(ancien_T2, Lc)
    
    # print(part_en_c)
    # print("les circos à fusionner")
    # print(c1)
    # print(c2)  

    # raise ZeroDivisionError
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

    nb_aretes_qui_coupent = len(cut_edge)
    if nb_aretes_qui_coupent in dico_nb_aretes_qui_coupent[0][1]:
        dico_nb_aretes_qui_coupent[0][1][nb_aretes_qui_coupent] +=1 
    else : 
        dico_nb_aretes_qui_coupent[0][1][nb_aretes_qui_coupent] =1 




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
    if nb_circos == 2 : 
        decoupage, moy_en_c = un_tour_de_Recom(liste_adj, decoupage,moy_en_c, Lc, epsilon, nb_circ)

    else : 

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


def genere_distribution(nb_decoupages, indice_debut, nb_tours, numDep, nb_circos, nomFichierEntree, nomFichierDecoupInit, dico_nb_aretes_qui_coupent): #, nomFichierEntree, nomFichierDecoupInit, nomFichierSortie):



    print("on se lance dans le genere_distribution ")

    # à changer si on n'est pas sur les mêmes fichiers .out !!
    nc, Lc, ns, Ls, decoup_init = lecture_bis(nomFichierEntree, nomFichierDecoupInit)
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)
    liste_adj = listeadj(mat_adj)
    # print(mat_adj)
    # print(liste_adj)
    epsilon = taux

    print("decoupage initial")
    print(decoup_init[indice_debut])

    # raise ZeroDivisionError

    monFichier=open(nomFichierSortie,"w")
    F = []
    for i in range(nb_decoupages):
        print("~~~~~~~~~~~~~~~~~~~~~~~~ " + str(i) + " ~~~~~~~~~~~~~~~~~~~~~~")
        F = (genere_un_decoupage_selon_la_decomp_recom(liste_adj, Lc, epsilon, decoup_init[indice_debut], nb_tours, nb_circos))
        F_prime = formalise_decoupage(F, nb_circos)
        card_coupe = calcule_card_de_la_coupe_pondere(F_prime, mat_adj)

        


        for li in F_prime:
            monFichier.write(str(li)+" ") 
        print(card_coupe)
        monFichier.write(str(card_coupe))
        monFichier.write("\n")

    monFichier.write("decoup_initiale :  " + str(formalise_decoupage(decoup_init[indice_debut], nb_circos)))

    # L = [0] * len(decoup_init)

    
  
    monFichier.close()
    print(nb_tours)
    
    print("a priori tout à été écrit dans le fichier : " + nomFichierSortie)
    dico_nb_aretes_qui_coupent = [(numDep,{})] + dico_nb_aretes_qui_coupent

    return dico_nb_aretes_qui_coupent



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

def algo_matrix_tree_theorem(mat_adj, liste_adj):
    lap_mat = mat_adj


    lap_mat = np.array(lap_mat)

    # ligne_col_supp = random.randint(len(lap_mat) -1 )
    ligne_col_supp = 0

    lap_mat = np.delete(lap_mat, ligne_col_supp, 0)
    lap_mat = np.delete(lap_mat, ligne_col_supp, 1)

    nb_spanning_tree = np.linalg.det(lap_mat)
    return nb_spanning_tree

def fait_le_tableau(numDep, nb_circos, nomFichierEntree, nomFichierDecoupInit ):

    #pour avoir la mat d'adj
    nc, Lc, ns, Ls, decoup_init = lecture_bis(nomFichierEntree, nomFichierDecoupInit)
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)
    liste_adj = listeadj(mat_adj)
    epsilon = taux




    mon_fichier = open(nomFichierSortie, "r")
    fichier_ecrire = "avec_dist_recom_pondere/tableau_sortie_Recom" + str(numDep) + ".txt"
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
        rr = formalise_decoupage(e, nb_circos)
        taille_coupe = calcule_card_de_la_coupe_pondere(rr, mat_adj)

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




def fait_histogramme(numDep):
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)
    liste_adj = listeadj(mat_adj)
    epsilon = taux




    fichier_ecrire = "avec_dist_recom_pondere/tableau_sortie_Recom" + str(numDep) + ".txt"
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
        taille_coupe_pondere = float(decoup_en_c[-1])
        if taille_coupe_pondere < 0:
            taille_coupe_pondere = (-1) * taille_coupe_pondere
        if taille_coupe_pondere in dico : 

            #format : découpage, nombre d'apparition du découpage
            dico[taille_coupe_pondere].append((decoup_en_c[:-3], (int(decoup_en_c[-3]))))
        else :
            dico[taille_coupe_pondere] = [(decoup_en_c[:-3], int(decoup_en_c[-3]))]



    print(dico)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    tab_ordonne, t_decoupages_associes, tab_taille_coupe_pondere = converti_en_tableau_de_tableaux(dico, 0)
        


    #bidouille : 
    pour_histo =[]
    for i in range(len(tab_ordonne)):
        pour_histo.append([])
        for j in range(len(tab_ordonne[i])):
            elt_bis= tab_ordonne[i][j]
            for cpt in range(elt_bis):
                pour_histo[i].append(tab_taille_coupe_pondere[j])


    
    print("tab ordonneeeeeeeeee")
    print(tab_ordonne)
    print("fin tab ordonneeee")
    #print(t_decoupages_associes)
    print(tab_taille_coupe_pondere)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs = np.random.randint(0, 10, 50)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs2 = np.random.randint(0, 10, 50)
    # # Création d'un tableau avec les intervalles centrés sur la valeur entière
    # inter = np.linspace(-0.5, 9.5, 11)

    plt.hist(pour_histo, bins=tab_taille_coupe_pondere, rwidth=0.8, stacked=True,
            label=['Valeurs 1', 'Valeurs 2'])  # Création de l'histogramme
    plt.xlabel('Taille de la coupe (regarder le nb à gauche)')
    plt.xticks(tab_taille_coupe_pondere)
    plt.ylabel('Nombre de découpage (une couleur = un découpage)')
    plt.title("Dpt : " + str(numDep) + " , 5%, Nombre de découpage en fct de la coupe")
    plt.legend()
    
    plt.savefig("graphiques/nb_occ_en_fct_taille_coupe_pondere/" + str(numDep) + ".png")
    plt.show()


def calc_produit_de_somme_de_tailles_circos(decoupage, ncirc, mat_adj):
    tab_tailles_circos = [(0, 0)] * ncirc 
    taille_coupe = (0,0)
    for i in range(len(mat_adj)):
        for j in range(i, len(mat_adj)):
            if mat_adj[i][j] > 0 and decoupage[i] == decoupage[j]:
                u = tab_tailles_circos[int(decoupage[i])] 

                tab_tailles_circos[int(decoupage[i])] = (u[0] + mat_adj[i][j], u[1]+1) 
            else : 
                taille_coupe = (taille_coupe[0] + mat_adj[i][j], taille_coupe[1]+1) 


    res = 1 
    for elt in tab_tailles_circos:
        res += (elt[0]/elt[1])

    return res, (taille_coupe[0]/taille_coupe[1])







def fait_histogramme_bis(numDep, ncirc):
    nc, Lc, ns, Ls, decoup_init = lecture_bis()
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)
    liste_adj = listeadj(mat_adj)
    epsilon = taux




    fichier_ecrire = "avec_dist_recom_pondere/tableau_sortie_Recom" + str(numDep) + ".txt"
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

        taille_produit_taille_circos, taille_coupe = calc_produit_de_somme_de_tailles_circos(decoup_en_c,ncirc, mat_adj )
        a_mettre_dans_dico = taille_produit_taille_circos/taille_coupe
        if a_mettre_dans_dico in dico : 

            #format : découpage, nombre d'apparition du découpage
            dico[a_mettre_dans_dico].append((decoup_en_c[:-3], (int(decoup_en_c[-3]))))
        else :
            dico[a_mettre_dans_dico] = [(decoup_en_c[:-3], int(decoup_en_c[-3]))]



    print(dico)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    tab_ordonne, t_decoupages_associes, tab_taille_coupe_pondere = converti_en_tableau_de_tableaux(dico, 0)
        


    #bidouille : 
    pour_histo =[]
    for i in range(len(tab_ordonne)):
        pour_histo.append([])
        for j in range(len(tab_ordonne[i])):
            elt_bis= tab_ordonne[i][j]
            for cpt in range(elt_bis):
                pour_histo[i].append(tab_taille_coupe_pondere[j])


    
    print("tab ordonneeeeeeeeee")
    print(tab_ordonne)
    print("fin tab ordonneeee")
    #print(t_decoupages_associes)
    print(tab_taille_coupe_pondere)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs = np.random.randint(0, 10, 50)
    # # Tableau de 50 valeurs comprises entre 0 et 9
    # valeurs2 = np.random.randint(0, 10, 50)
    # # Création d'un tableau avec les intervalles centrés sur la valeur entière
    # inter = np.linspace(-0.5, 9.5, 11)

    plt.hist(pour_histo, bins=tab_taille_coupe_pondere, rwidth=0.8, stacked=True,
            label=['Valeurs 1', 'Valeurs 2'])  # Création de l'histogramme
    plt.xlabel('Taille de la coupe (regarder le nb à gauche)')
    plt.xticks(tab_taille_coupe_pondere)
    plt.ylabel('Nombre de découpage (une couleur = un découpage)')
    plt.title("Dpt : " + str(numDep) + " , 5%, Nombre de découpage en fct de la coupe")
    plt.legend()
    
    plt.savefig("graphiques/nb_occ_en_fct_prod_taille_circos/" + str(numDep) + ".png")
    plt.show()


#c'est une fonction qui crée un tableau de tableau qui représente le dico passé en param
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

def explore_st(mat_adj, s, vus, st, arbre_en_c, decoup_en_c, circo_en_c ):
    if not (-2) in arbre_en_c:
        st.append(arbre_en_c)
        return st
    else : 
        for i in range(len(mat_adj)):
            if decoup_en_c[i] == circo_en_c and mat_adj[s][i] > 0 and not vus[i] :
                arbre_bis = arbre_en_c.copy()
                arbre_bis[i] = s
                print(arbre_bis)
                print(arbre_en_c)
                vus[i] = True
                st = explore_st(mat_adj, i, vus, st, arbre_bis, decoup_en_c, circo_en_c)
                vus[i] = False
        return st

####
#pour faire graphique nb occ en fonction de nb st




def algo_matrix_tree_theorem(mat_adj, liste_adj):
    lap_mat = mat_adj


    lap_mat = np.array(lap_mat)

    # ligne_col_supp = random.randint(len(lap_mat) -1 )
    ligne_col_supp = 0

    lap_mat = np.delete(lap_mat, ligne_col_supp, 0)
    lap_mat = np.delete(lap_mat, ligne_col_supp, 1)

    nb_spanning_tree = np.linalg.det(lap_mat)
    return nb_spanning_tree

# mat_adj = [[0,1,1,1], [1,0,1,0], [1,1,0,1], [1,0,1,0]]
# lap_mat = [[3,-1, -1, -1], [-1, 2, -1, 0], [-1, -1, 3, -1], [-1, 0, -1, 2]]
# print(algo_matrix_tree_theorem(lap_mat, []))

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
                if elt_un != elt_deux and mat_adj[elt[elt_un]][elt[elt_deux]] > 0:
                    mat_adj_bis[elt_un][elt_deux] = -1 
                    # mat_adj_bis[elt_deux][elt_un] = -1 
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

mat_adj = [[0,1,1,1,1,0,0,1], 
           [1,0,1,0,0,0,0,0], 
           [1,1,0,1,0,0,0,0],
           [1,0,1,0,0,0,0,0],
           [1,0,0,0,0,1,1,1],
           [0,0,0,0,1,0,1,0],
           [0,0,0,0,1,1,0,1],
           [1,0,0,0,1,0,1,0]]
CC = [[0,1,2,3], [4,5,6,7]]

print(calcule_nombre_de_foret_couvrante(mat_adj, CC))

def fait_graph_occ_en_fct_de_nb_st(numDep, nomFichierEntree, nomFichierDecoupInit, nb_circos):
    nc, Lc, ns, Ls, decoup_init = lecture_bis(nomFichierEntree, nomFichierDecoupInit)
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)
    liste_adj = listeadj(mat_adj)
    epsilon = taux




    fichier_ecrire = "avec_dist_recom_pondere/tableau_sortie_Recom" + str(numDep) + ".txt"
    fichier_tableau = open(fichier_ecrire, "r")
    decoupages_tableau = fichier_tableau.readlines()
    # a_supprimer = int(decoupages_possibles_avant[1].split()[0])
    # decoupages_possibles = decoupages_possibles_avant[2:] 
    nb_decoupages = int(decoupages_tableau[1])
    fichier_tableau.close()

    # decoupages_tableau = ["", "", "0 0 0 0 1 1 1 1 3 3 3"]
    # nb_decoupages = 1

    
    dico = {}

    for i in range(2, nb_decoupages+2):
        decoup_en_c = decoupages_tableau[i].split()

        #on definit CC pour faire calculer le nombre de forets couvrantes : 
        CC = []
        for i in range(nb_circos):
            CC.append([])
        for j in range(len(decoup_en_c) - 3):
            CC[int(decoup_en_c[j])].append(j)

        # mat_adj = [[0,1,1,1,1,0,0,1], 
        #             [1,0,1,0,0,0,0,0], 
        #             [1,1,0,1,0,0,0,0],
        #             [1,0,1,0,0,0,0,0],
        #             [1,0,0,0,0,1,1,1],
        #             [0,0,0,0,1,0,1,0],
        #             [0,0,0,0,1,1,0,1],
        #             [1,0,0,0,1,0,1,0]]
        # print(CC)
        # print(mat_adj)
        # print("``````````````")
        #nb_f doit contenir le nombre de st du découpage en question 
        nb_f = calcule_nombre_de_foret_couvrante(mat_adj, CC)
        if nb_f == 3 : 
            print("[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]")
            print(i)
            print(decoup_en_c)
            print("[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]")

        # print(nb_f)
        # raise ZeroDivisionError

        # pour faire nb_occ en fct de nb_st
        # if int(decoup_en_c[-3]) > 0 :
        #     if nb_f in dico : 

        #         #format : découpage, nombre d'apparition du découpage
        #         dico[nb_f].append((decoup_en_c[:-3], int(decoup_en_c[-3])))
        #     else :
        #         dico[nb_f] = [(decoup_en_c[:-3], int(decoup_en_c[-3]))]

        # pour faire nb_st en fct de nb_occ
        if int(decoup_en_c[-3]) > 0 :
            if int(decoup_en_c[-3]) in dico : 

                #format : découpage, nombre d'apparition du découpage
                dico[int(decoup_en_c[-3])].append((decoup_en_c[:-3], nb_f))
            else :
                dico[int(decoup_en_c[-3])] = [(decoup_en_c[:-3], nb_f)]



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
    # plt.savefig("graphiques/nb_occ_en_fct_nb_spanning_forest_pondere/" + str(numDep) + ".png")
    plt.savefig("graphiques/nb_occ_en_fct_nb_spanning_forest_pondere/nb_st_en_fct_nb_occ" + str(numDep) + ".png")

     




###########################################################################

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, rank, x, y):
    rx, ry = find(parent, x), find(parent, y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[ry] < rank[rx]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True

def enumerate_spanning_trees(adj, decoup_en_c, ind_en_c, edges):
    n = len(adj)
    if n == 0:
        return []
    

    results = []

    #count = nb arête dans l'arbre en construction
    def backtrack(index, count, uf_parent, uf_rank, tree_parent, ind_en_c):
        if count == n - 1:
            results.append(tree_parent[:])
            return
        #si on abouti a un truc qui n'est pas un arbre
        if index == len(edges):
            return

        u, v = edges[index]
        
        if decoup_en_c[u] == str(ind_en_c) and decoup_en_c[v] == str(ind_en_c):
            # print("edge_en_c : ")
            # print(u)
            # print(v)
            # print("combien d'aretes : ")
            # print(count)
            pu = find(uf_parent, u)
            pv = find(uf_parent, v)

            # dans tous les cas on backtrack en supposant qu'on ne met pas l'arête en cours, 
            # si de plus ça ne forme pas un cycle, on backtrack en essayant de mettre l'arête dans l'arbre
            if pu != pv:
                new_uf_parent = uf_parent[:]
                new_uf_rank = uf_rank[:]
                if union(new_uf_parent, new_uf_rank, u, v):
                    new_tree_parent = tree_parent[:]
                    # Fixation du parent: choisissons par convention le sommet de plus petit indice comme parent
                    # print("on prend l'arete : " + str(u) + "  " + str(v))
                    # print(new_tree_parent)
                    if new_tree_parent[v] == -1:
                        new_tree_parent[v] = u
                    elif new_tree_parent[u] == -1:
                        new_tree_parent[u] = v
                    else : 
                        # cf le papuer que hj'ai écrit sur une feuille volante dans la pochette orange
                        # on fait toujours la situation c > b car on n'a pas d'invariant la dessus finalement
                        # u = a, b = v
                        
                        new_tree_parent[v] = u
                        noeud_en_c = v
                        while tree_parent[noeud_en_c] != -1 : 
                            new_tree_parent[tree_parent[noeud_en_c]] = noeud_en_c
                            noeud_en_c = tree_parent[noeud_en_c]
                    #     print("les deux noeuds ont déja des parents ! ")
                        
                    # print(new_tree_parent)

                    backtrack(index + 1, count + 1, new_uf_parent, new_uf_rank, new_tree_parent, ind_en_c)

            backtrack(index + 1, count, uf_parent, uf_rank, tree_parent, ind_en_c)
        else :
            backtrack(index + 1, count, uf_parent, uf_rank, tree_parent, ind_en_c)
            
    uf_parent = list(range(n))
    uf_rank = [0] * n
    tree_parent = [-1] * n  # Tous init à -1, racine aura -1
    dico_tailles = {}

    n_moins_nb_a_atteindre = 0
    for i in range(len(decoup_en_c)):
        
        if str(ind_en_c) != decoup_en_c[i]:
            n_moins_nb_a_atteindre += 1
            tree_parent[i] = -2
    # print(tree_parent)

    backtrack(0, n_moins_nb_a_atteindre, uf_parent, uf_rank, tree_parent, ind_en_c)
    return results

def trouve_st_des_circos(g, decoup_en_c, nb_c):
    edges = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            if g[i][j]:
                edges.append((i, j))



    res = []
    for i in range(nb_c):
        st_de_la_circo = enumerate_spanning_trees(g, decoup_en_c, i, edges)
        res.append(st_de_la_circo)
    return res





###########################################################################
def calcule_les_poids_de_tous_les_sts(mat_adj, decoup_en_c, nb_circos, dico_pond):
    sts = trouve_st_des_circos(mat_adj, decoup_en_c, nb_circos)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(sts)
    # print("#################")


    poids_sts = []
    #pour chaque circo
    for elt in sts :
        poids_elt = []
        #on parcours les arbres couvrants de la circo en cours
        for st in elt : 
            poids_arbre = 0
            
            
            for i in range(len(st)) : 
                if st[i] >= 0 : 
                    if (i, st[i]) in dico_pond:
                        poids_arbre += dico_pond[(i, st[i])]
                    elif (st[i], i) in dico_pond:
                        poids_arbre += dico_pond[( st[i], i)]
                    else : 
                        print("##################")
                        print(i)
                        print(st[i])
                        print("eeruer, il semblerait qu'on ait emprunté une arête qui n'existe pas (car pas pondérée)")
                        raise ZeroDivisionError
            poids_elt.append(poids_arbre)
        poids_sts.append(poids_elt)

    res = poids_sts[0]
    for i in range(1,len(poids_sts)):
        res_bis = []
        for j in range(len(poids_sts[i])):
            
            for elt in res : 
                # ici c'est bien un " + " normalement
                res_bis.append(elt + poids_sts[i][j])

        res = res_bis

    return res



# ATTTENTION : utilise lecture bis donc a priori le fichier d'entrée avec les 5 lignes donnant les noms de fichiers est utile !!
# en fait non 
def fait_histo_nb_st_en_fct_card_coupe_pondere(numDep, nb_circos, nomFichierEntree, nomFichierDecoupInit):
    dico_ponderations = calcule_les_ponderations(numDep)

    mon_fichier = open(nomFichierDecoupInit, "r")
    lignes = mon_fichier.readlines()
    mon_fichier.close()

    nc, Lc, ns, Ls, decoup_init = lecture_bis(nomFichierEntree, nomFichierDecoupInit)
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)

    a_supp = int(lignes[1].split()[0])
    nb_dec = int(lignes[1].split()[1])

    poids_des_coupes = []
    poids_spanning_tree = []
    dico = {}

    for i in range(nb_dec):
        print("~~~~~~~~~~~~~~~~~~~~~~ " + str(i+1) + " / " + str(nb_dec) + "~~~~~~~~~~~~~~~~~")
        decoupage_en_c = lignes[i+2].split()[a_supp:]
        # print("decoupp en ccc")
        # print(decoupage_en_c)

        liste_poids_des_sts = calcule_les_poids_de_tous_les_sts(mat_adj, decoupage_en_c, nb_circos, dico_ponderations)

        taille_de_la_coupe = 0
        for j in range(len(decoupage_en_c)):
            for k in range(len(decoupage_en_c)):
                if k != j : 
                    if (j,k) in dico_ponderations:
                        if decoupage_en_c[j] != decoupage_en_c[k]:
                            taille_de_la_coupe += dico_ponderations[(j,k)]
                    elif (k,j) in dico_ponderations:
                        if decoupage_en_c[j] != decoupage_en_c[k]:
                            taille_de_la_coupe += dico_ponderations[(k,j)]
                    
                
        #si on veut afficher les poids de tous les ST pour un poids de coupe donné
        # poids_des_coupes += [taille_de_la_coupe] * len(liste_poids_des_sts)
        # poids_spanning_tree += liste_poids_des_sts

        #si on veut afficher seulement la moyenne
        # poids_des_coupes += [taille_de_la_coupe]
        # poids_spanning_tree += [np.mean(liste_poids_des_sts)]

        #si on veut afficher seulement le max pour chaque taille de coupe
        poids_des_coupes += [taille_de_la_coupe]
        poids_spanning_tree += [max(liste_poids_des_sts)]

    # print(poids_des_coupes)
    # print(poids_spanning_tree)
    plt.plot(poids_des_coupes,poids_spanning_tree, 'ro')
    plt.title("Département " + str(numDep))
    plt.xlabel('poids des coupes')
    plt.ylabel('poids_des_sts')
    plt.savefig("graphiques/poids_st_en_fct_poids_coupe/dpt_" + str(numDep) + ".png")
    plt.show()


def coupling_from_the_past(numDep, nb_circos, nomFichierEntree, nomFichierDecoupInit):

    #pour avoir la mat d'adj
    nc, Lc, ns, Ls, decoup_init = lecture_bis(nomFichierEntree, nomFichierDecoupInit)
    print("combien de tours :   ") # c'est la dernière ligne du fichier d'entrée (juste après le pourcentage d'erreur)
    mat_adj = graphadj(nc, Lc, ns, Ls, numDep)
    liste_adj = listeadj(mat_adj)
    epsilon = taux


    # mon_fichier = open(nomFichierSortie, "r")
    # fichier_ecrire = "avec_dist_recom_pondere/tableau_sortie_Recom" + str(numDep) + ".txt"

    #pour récupérer tous les découpages possibles
    fichier_source = open(nomFichierDecoupInit, "r")
    # fichier_sortie = open(fichier_ecrire, "w")
    decoupages_possibles_avant = fichier_source.readlines()
    a_supprimer = int(decoupages_possibles_avant[1].split()[0])
    decoupages_possibles = decoupages_possibles_avant[2:]
    fichier_source.close()

    # lignes = mon_fichier.readlines()
    # mon_fichier.close()
    dico = {}

    
    for elt in decoupages_possibles : 
        #si les decoupages ne sont pas encore normalisés : 
        
        e = elt.split()[a_supprimer:]

        rr = formalise_decoupage(e, nb_circos)
        #normalement rr est sous la forme d'entrée et de sortie de "un tour de Recom"
        print(rr)

        assert(rr not in dico)
        dico[rr] = rr

    moy_en_c = moyenne_initiale(dico[dico.keys()[0]], nb_circos, Lc)
    


    while len(dico.values()) > 1: 
        dico_bis = {}
        for elt in dico : 
            dico_bis[elt] = dico[un_tour_de_Recom(liste_adj, elt, moy_en_c, Lc,epsilon, nb_circos)]
        dico = dico_bis

    return dico[dico.keys()[0]]






liste_departements_qui_nous_interessent = [(5,2), (32,2),
                                           (12,3), (10,3)]
#                                         #    (24,4), (50,4),
#                                         #    (21,5), (22,5)]
liste_departements_qui_nous_interessent_20_pour_cent = [(68,6), (74,6),
                                                        (83,8), (57,9), (31,10), (94,11)]

dico_nb_aretes_qui_coupent = [(111111, {})]
# liste_departements_qui_nous_interessent_20_pour_cent = []
# liste_departements_qui_nous_interessent = [ (50,4)]
for (numDep, nb_circos) in liste_departements_qui_nous_interessent:
    nomFichierEntree = "graphe_depts/"+ str(numDep) + ".out"

    nomFichierSortie = "avec_dist_recom_pondere/sortie_Recom" + str(numDep) + ".txt"
    if numDep < 10 : 

        nomFichierDecoupInit = "actuel-5pour100/sortie0"+ str(numDep) + ".txt"
    else : 
        nomFichierDecoupInit = "actuel-5pour100/sortie"+ str(numDep) + ".txt"

    # CHANGGGGGGGEEEEEERRRRRR
    taux = 0.05
    # CHANGGGGGGGEEEEEERRRRRR
    # dico_nb_aretes_qui_coupent = genere_distribution(1000, 0, 1000, numDep, nb_circos, nomFichierEntree, nomFichierDecoupInit, dico_nb_aretes_qui_coupent)

    # fait_le_tableau(numDep, nb_circos, nomFichierEntree, nomFichierDecoupInit)
    fait_graph_occ_en_fct_de_nb_st(numDep, nomFichierEntree, nomFichierDecoupInit, nb_circos)
    print("||||||||||||||||||||||||||||||||||||||||||||||||")
    print(dico_nb_aretes_qui_coupent)
    print("||||||||||||||||||||||||||||||||||||||||||||||||")


for (numDep, nb_circos) in liste_departements_qui_nous_interessent_20_pour_cent:
    nomFichierEntree = "graphe_depts/"+ str(numDep) + ".out"

    nomFichierSortie = "avec_dist_recom_pondere/sortie_Recom" + str(numDep) + ".txt"
    if numDep < 10 : 

        nomFichierDecoupInit = "actuel-20pour100/sortie0"+ str(numDep) + ".txt"
    else : 
        nomFichierDecoupInit = "actuel-20pour100/sortie"+ str(numDep) + ".txt"
    taux = 0.2
    # dico_nb_aretes_qui_coupent = genere_distribution(1000, 0, 1000, numDep, nb_circos, nomFichierEntree, nomFichierDecoupInit, dico_nb_aretes_qui_coupent)
    fait_graph_occ_en_fct_de_nb_st(numDep, nomFichierEntree, nomFichierDecoupInit, nb_circos)
    print("||||||||||||||||||||||||||||||||||||||||||||||||")
    print(dico_nb_aretes_qui_coupent)
    print("||||||||||||||||||||||||||||||||||||||||||||||||")




#nb_samples, indice decoup de départ, nb tours, numDep
# genere_distribution(100, 1, 100, numDep, nb_circos)
# fait_le_tableau(numDep, nb_circos)
# fait_histogramme(numDep)

# fait_histogramme_bis(numDep, ncirc)

# mat_adj = [[0,1,0,1], [1,0,1,1], [0,1,0,1], [1,1,1,0]]
# print(calcule_tous_les_st(mat_adj, nomFichierEntree, nomFichierDecoupInit))



    
