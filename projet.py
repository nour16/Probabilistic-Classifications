# -*- coding: utf-8 -*-
import math
import utils
import scipy
import numpy as np
import pandas as pd 
from matplotlib import pyplot

def getPrior(df):
    """
    Cette fonction rend un dictionnaire dis_res contenant l'estimation de la probabilité 
    à priori de la classe 1 et l'intervalle de confiance(min5pourcnt et max5pourcant) à 95% pour l'estimation de cette probabilite 
    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
    
    """
    sick=df[(df['target']==1)]
    dict_res=dict()
    p=(len(sick)*1.)/(len(df))
    dict_res['estimation']=p
    ecart_type=math.sqrt(p*(1-p)/len(df))
    m=p
    dict_res['min5pourcent']=m - 1.96*(ecart_type)
    dict_res['max5pourcent']=m + 1.96*(ecart_type)
    return dict_res



class APrioriClassifier(utils.AbstractClassifier):
    """
    Un classifeur enfant de AbstractClassifier qui implémente un algorithme 
    estimant la classe de chaque individu par la classe majoritaire
    """
    def __init__(self):
        """
        initialise les paramètres de la classe et estime la classe
        majoritaire de chaque individu a partir d'un dictionnaire 
        renvoye par le fonction getPrior(df)
        """
        self.train=pd.read_csv("train.csv") 
        self.prior=getPrior(self.train)
        if(self.prior['estimation']>=0.5):
            self.classMaj=1
        else:
            self.classMaj=0
    
    def estimClass(self,attrs):
        """
        retourne la classe éstimée 0 ou 1 à partir d'un dictionnaire d'attributs 'attrs'
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return la classe 0 ou 1
        """
        return self.classMaj

    def statsOnDF(self,df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs 
        de classification et rend un dictionnaire.
        :param df:  le dataframe à tester
        :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        """
        dico=dict()
        dico['VP']=0
        dico['VN']=0
        dico['FP']=0
        dico['FN']=0
        for i in range(len(df)):
            dic=utils.getNthDict(df,i)
            t_prevue=self.estimClass(dic)
            target=dic['target']
            if(target==1 and t_prevue==1):
                dico['VP']+=1
            elif(target==0 and t_prevue==0):
                dico['VN']+=1
            elif(target==0 and t_prevue==1):
                dico['FP']+=1
            else:
                dico['FN']+=1
        dico['Précision']=dico['VP']*1./(dico['VP']+dico['FP'])
        dico['Rappel']=dico['VP']*1./(dico['VP']+dico['FN'])
        return dico

#-------------Q3.a-----------------#
def P2D_l(df,attr):
    """
    Rend un dictionnaire associant aux valeurs t=0 et t=1 un dictionnaire associant 
    à la valeur 'a' de l'attribut 'attr' sa proba P(attr=a|target=t)
    :param df:le dataframe à tester
    :param attr:nom de l'attribut 
    """
    dico=dict()
    dico[1]=dict()
    dico[0]=dict()
    tmp1=dict()
    tmp0=dict()
    p0=len(df[(df['target']==0)])/len(df) 
    p1=1-p0
    ens_val_attr={df[attr][i] for i in range(len(df))}# ensemble des valeurs de l'attribut attr
    for a in ens_val_attr:
        p_inter_t1=len(df[(df['target']==1) & (df[attr]==a)])/len(df)
        p_inter_t0=len(df[(df['target']==0) & (df[attr]==a)])/len(df)
        tmp1[a]=p_inter_t1/p1
        tmp0[a]=p_inter_t0/p0
    dico[1]=tmp1
    dico[0]=tmp0
    return dico

def P2D_p(df,attr):
    """
    Rend un dictionnaire associant à la valeur a de attr un dictionnaire 
    associant à t=0 ou t=1 la probabilité P(target=t|attr=a)
    :param df:le dataframe à tester
    :param attr:nom de l'attribut 
    """
    dico=dict()
    ens_val_attr={df[attr][i] for i in range(len(df))}
    for a in ens_val_attr:
        pa=len(df[(df[attr]==a)])/len(df)
        dico[a]=dict()
        tmp=dict()
        p_inter_t1=len(df[(df['target']==1) & (df[attr]==a)])/len(df)
        p_inter_t0=len(df[(df['target']==0) & (df[attr]==a)])/len(df)
        tmp[1]=p_inter_t1/pa
        tmp[0]=p_inter_t0/pa
        dico[a]=tmp
    return dico

class ML2DClassifier(APrioriClassifier):
    """
    classifieur qui implémente un algorithme estimant la classe target d'un
    individu en utilisant le principe de maximum de vraissemblance"""

    def __init__(self,df,attr):
        """
        :param df:le dataframe à tester
        :param attr:nom de l'attribut 
        """
        self.df=df
        self.attr=attr
        self.dico_proba=P2D_l(self.df,self.attr)

    def estimClass(self,dico_individu):
        """"
        :param dico_individu:dictionnaire contenant tous les individus et leurs attributs
        :return:la valeur de la classe ayant le maximum de vraisemblance après l'avoir 
        calculer dans le constructeur
        """
        val_a=dico_individu[self.attr]
        if(self.dico_proba[1][val_a]>self.dico_proba[0][val_a]):
            return 1
        else : 
            return 0

class MAP2DClassifier(APrioriClassifier):
    """
    classifeur qui implémente un algorithme estimant la classe target d'un
    individu en utilisant le principe du maximum a posteriori"""

    def __init__(self,df,attr):
        """
        :param df:le dataframe à tester
        :param attr:nom de l'attribut 
        """
        self.df=df
        self.attr=attr
        self.dico_proba=P2D_p(self.df,self.attr)

    def estimClass(self,dico_individu):
        """"
        :param dico_individu:dictionnaire contenant tous les individus et leurs attributs
        :return:la valeur de la classe avec la plus grande probabilité de l'attribut attr de la classe
        """
        val_a=dico_individu[self.attr]
        if(self.dico_proba[val_a][1]>self.dico_proba[val_a][0]):
            return 1
        else : 
            return 0

#---------------------------Q4---------------------------------#
def nbParams(df,liste_attrs=[]):
    """
        affiche la taille mémoire nécessaire pour représenter la table des 
        probabilités P(target|att1...attK)
        On supposse qu'un float est représenté sur 8octets 
        Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        liste_attrs: liste attributs 
    """
    taille_memoire=0
    if(len(liste_attrs)==0):
        for a in df:
            liste_attrs.append(a)
    nb_var=len(liste_attrs)     
    taille_memoire=1
    for a in liste_attrs:
        taille_memoire=taille_memoire*len(P2D_p(df,a))
    taille_memoire=taille_memoire*8
    #affichage
    if(taille_memoire<2**10):
        print("#{} variable(s) : {} octets".format(nb_var,taille_memoire))
        return
    else:
        if(taille_memoire>=(2**10) and taille_memoire<(2**20)):
            k=taille_memoire//(2**10)#kilos octets
            r=taille_memoire-k*(2**10)#octets
            print("#{} variable(s) : {} octets = {}ko {}o".format(nb_var,taille_memoire,k,r))
            return 
        if(taille_memoire<2**30 and taille_memoire>=2**20):
            m=taille_memoire//(2**20)#megas octets
            r1=taille_memoire-m*(2**20)
            k=r1//(2**10)#kilos octets
            r2=r1-k*(2**10)
            print("#{} variable(s) : {} octets = {}mo {}ko {}o".format(nb_var,taille_memoire,m,k,r2))
            return 
        else:
            g=taille_memoire//(2**30)#giga octets
            r=taille_memoire-g*(2**30)
            m=r//(2**20)#megas octets
            r1=r-m*(2**20)
            k=r1//(2**10)#kilos octets
            r2=r1-k*(2**10)
            print("#{} variable(s) : {} octets = {}go {}mo {}ko  {}o".format(nb_var,taille_memoire,g,m,k,r2))
            return 

def nbParamsIndep(df):
    """
        affiche la taille mémoire nécessaire pour représenter les tables de
        probabilité en supposant l'indépendance des variables
        On supposse qu'un float est représenté sur 8octets 
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
      
    """
    taille_memoire=0
    nb_var=0
    for a in utils.getNthDict(df,0):
        taille_memoire=taille_memoire+len(P2D_p(df,a))
        nb_var+=1
    taille_memoire=taille_memoire*8
    #affichage
    if(taille_memoire<2**10):
        print("#{} variable(s) : {} octets".format(nb_var,taille_memoire))
        return
    else:
        if(taille_memoire>=(2**10) and taille_memoire<(2**20)):
            k=taille_memoire//(2**10)#kilos octets
            r=taille_memoire-k*(2**10)#octets
            print("#{} variable(s) : {} octets = {}ko {}o".format(nb_var,taille_memoire,k,r))
            return 
        if(taille_memoire<2**30 and taille_memoire>=2**20):
            m=taille_memoire//(2**20)#megas octets
            r1=taille_memoire-m*(2**20)
            k=r1//(2**10)#kilos octets
            r2=r1-k*(2**10)
            print("#{} variable(s) : {} octets = {}mo {}ko {}o".format(nb_var,taille_memoire,m,k,r2))
            return 
        else:
            g=taille_memoire//(2**30)#giga octets
            r=taille_memoire-g*(2**30)
            m=r//(2**20)#megas octets
            r1=r-m*(2**20)
            k=r1//(2**10)#kilos octets
            r2=r1-k*(2**10)
            print("#{} variable(s) : {} octets = {}go {}mo {}ko  {}o".format(nb_var,taille_memoire,g,m,k,r2))
            return

#---------------------------------q5---------------#
def drawNaiveBayes(df,attr):
    """
        fonction qui dessine le graphe des dépendances des attributs selon le modèle naive bayes
        à partir d'un dataframe et du nom de la colonne 
        Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            attr:le parent de tous les autres attributs
    """
    ch_dep=""
    for a in df.keys():
        if(a!=attr):
            ch_dep+=attr+"->"+a+";"
    return utils.drawGraph(ch_dep)



def nbParamsNaiveBayes(df,attr,liste_attrs=['target','age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']):
    """
       calcule la taille mémoire nécessaire pour représenter les tables de probabilité 
       en utilisant l'hypothèse du Naive Bayes
        Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        attr : l'attribut dont dépondent tous les attributs de la liste_attrs
        liste_attrs: liste attributs dont on veut calculer la taille memoire des tables de probabilité 
    """
    taille_memoire=0
    

    nb_var=len(liste_attrs) 
    ens_val_attr={df[attr][i] for i in range(len(df[attr]))}
    if(len(liste_attrs)==0 or liste_attrs[0]==attr):
        taille_memoire=len(ens_val_attr)
    
    for a in liste_attrs: 
        if(attr!=a):
            ens_val_a={df[a][i] for i in range(len(df[a]))}
            taille_memoire=taille_memoire+len(ens_val_a)*len(ens_val_attr)
    taille_memoire=taille_memoire*8
    #affichage
    if(taille_memoire<2**10):
        print("#{} variable(s) : {} octets".format(nb_var,taille_memoire))
        return
    else:
        if(taille_memoire>=(2**10) and taille_memoire<(2**20)):
            k=taille_memoire//(2**10)#kilos octets
            r=taille_memoire-k*(2**10)#octets
            print("#{} variable(s) : {} octets = {}ko {}o".format(nb_var,taille_memoire,k,r))
            return 
        if(taille_memoire<2**30 and taille_memoire>=2**20):
            m=taille_memoire//(2**20)#megas octets
            r1=taille_memoire-m*(2**20)
            k=r1//(2**10)#kilos octets
            r2=r1-k*(2**10)
            print("#{} variable(s) : {} octets = {}mo {}ko {}o".format(nb_var,taille_memoire,m,k,r2))
            return 
        else:
            g=taille_memoire//(2**30)#giga octets
            r=taille_memoire-g*(2**30)
            m=r//(2**20)#megas octets
            r1=r-m*(2**20)
            k=r1//(2**10)#kilos octets
            r2=r1-k*(2**10)
            print("#{} variable(s) : {} octets = {}go {}mo {}ko  {}o".format(nb_var,taille_memoire,g,m,k,r2))
            return 

class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur qui implémente un algorithme estimant la classe d'un individu 
    en utilisant le maximum de vraissemblance et l'hypothèse de Naive Bayes 
    """

    def __init__(self,df):
        """
        :param df:le dataframe à tester
        """
        self.df=df
        self.dico_proba_a_sachant_t=dict()
        for a in utils.getNthDict(df,0):
            self.dico_proba_a_sachant_t[a]=P2D_l(df,a)

    def estimProbas(self,dico_individu):
        """ calcule la vraisemblance en utilisant l hypothese du Naive Bayes.
        :param:dico_individu: dictionnaire contenant les informations d'un individu
        :return:un dictionnaire associant à chacune des classes 0 et 1 sa vraissemblance """
        #dictionnaire contenant la vraisemblance pour la classe 0 et 1
        dico_v={0:1,1:1}
        for a,val_a in dico_individu.items():#pour avoir les noms des attributs
            if(a!="target"):
                tmp=self.dico_proba_a_sachant_t[a]
                if val_a in tmp[1]:
                    dico_v[1]=dico_v[1]*tmp[1][val_a]
                else:
                    dico_v[1]=dico_v[1]*0
                if val_a in tmp[0]:
                    dico_v[0]=dico_v[0]*tmp[0][val_a]
                else:
                   dico_v[0]=dico_v[0]*0
        return dico_v

    def estimClass(self,dico_individu):
        """"
        retourne la valeur de la classe ayant le maximum de vraisemblance
        :param: dico_individu:dictionnaire contenant tous les attributs d'un individu
        """ 
        dico_v=self.estimProbas(dico_individu)
        if(dico_v[0]<dico_v[1]):
            return 1
        else:
            return 0

#vraisemblance*p(t)/p(a1,a2...)  
class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur qui implémente un algorithme estimant la classe d'un individu 
    en utilisant le maximum à posteriori et l'hypothèse de Naive Bayes 
    """
    def __init__(self,df):
        """
        :param df:le dataframe à tester
        """
        self.df=df
        self.dico_proba_a_sachant_t=dict()
        self.p_t_1=getPrior(self.df)['estimation']#proba t=1
        self.p_t_0=1-self.p_t_1#proba t=0
        for a in utils.getNthDict(df,0):
            self.dico_proba_a_sachant_t[a]=P2D_l(df,a)
        
    
    def estimProbas(self,dico_individu):
        """ calcule l'estimation à postriori en utilisant l hypothese du Naive Bayes.
        :param:dico_individu: dictionnaire contenant les informations d'un individu
        :return:un dictionnaire associant à chacune des classes 0 et 1 l'estimation à posteriori 
        """
        #def estimProbas(self,dico_individu):
        dico_v={0:1,1:1}
        for a,val_a in dico_individu.items():#pour avoir les noms des attributs
            if(a!="target"):
                tmp=self.dico_proba_a_sachant_t[a]
                if val_a in tmp[1]:
                    dico_v[1]=dico_v[1]*tmp[1][val_a]
                else:
                    dico_v[1]=dico_v[1]*0
                if val_a in tmp[0]:
                    dico_v[0]=dico_v[0]*tmp[0][val_a]
                else:
                   dico_v[0]=dico_v[0]*0
                   
        dico_v[0]=dico_v[0]*self.p_t_0
        dico_v[1]=dico_v[1]*self.p_t_1
        p_observation=dico_v[0]+dico_v[1]
        if(p_observation!=0):
            dico_v[0]=dico_v[0]/p_observation
            dico_v[1]=dico_v[1]/p_observation

        return dico_v
    
    def estimClass(self,dico_individu):
        """"
        retourne la valeur de la classe ayant le maximum à postriori
        :param: dico_individu:dictionnaire contenant tous les attributs d'un individu
        """ 
        dico_v=self.estimProbas(dico_individu)
        if(dico_v[0]<dico_v[1]):
            return 1
        else:
            return 0

#----------------------------------------Q6----------------------------#
def isIndepFromTarget(df,attr,x):
    """ 
    Renvoie true si l'attribut attr est indépendant de target au sueil de x% et false sinon.
    Arguments:
       df {pandas.dataframe} -- le pandas.dataframe contenant les données
       attr: attribut sur lequel on va effectuer de le test d'independance
       x : le sueil d'independance d'un attribut
    """
    table_contin=pd.crosstab(df['target'],df[attr],margins=False)#creer la table de contingence
    v1, p, v2, v3 = scipy.stats.chi2_contingency(np.array(table_contin))
    if p > x:
        return True
    return False

class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur qui implémente un algorithme estimant la classe d'un individu en utilisant le maximum 
    de vraissemblance et l'hypothèse de Naive Bayes, optimisé avec les tests d'indépendances au seuil de x%
    """
    def __init__(self,df,x):
        """
        :param df:le dataframe à tester
        :param x : le sueil d'independance d'un attribut
        """
        self.df=df
        self.dico_proba_a_sachant_t=dict()
        for a in utils.getNthDict(df,0):
            if(not isIndepFromTarget(self.df,a,x)):#si l'attribut n'est pas independant on le prend en compte sinon on ne calcule pas sa probabilité
                self.dico_proba_a_sachant_t[a]=P2D_l(df,a)
        self.liste_ags_reduits=[i for i in self.dico_proba_a_sachant_t.keys()] #creer la liste des agruments non independants

    def estimProbas(self,dico_individu):
        """ calcule la vraisemblance en utilisant l hypothese du Naive Bayes.
        :param:dico_individu: dictionnaire contenant les informations d'un individu
        :return:un dictionnaire associant à chacune des classes 0 et 1 sa vraissemblance """
        #dictionnaire contenant la vraisemblance pour la classe 0 et 1
        dico_v={0:1,1:1}
        for a,val_a in dico_individu.items():#pour avoir les noms des attributs
            if a in self.liste_ags_reduits:
                if(a!="target"):
                    tmp=self.dico_proba_a_sachant_t[a]
                    if val_a in tmp[1]:
                        dico_v[1]=dico_v[1]*tmp[1][val_a]
                    else:
                        dico_v[1]=dico_v[1]*0
                    if val_a in tmp[0]:
                        dico_v[0]=dico_v[0]*tmp[0][val_a]
                    else:
                        dico_v[0]=dico_v[0]*0
        return dico_v

    def estimClass(self,dico_individu):
        """"
        retourne la valeur de la classe ayant le maximum de vraisemblance
        :param: dico_individu:dictionnaire contenant tous les attributs d'un individu
        """ 
        dico_v=self.estimProbas(dico_individu)
        if(dico_v[0]<dico_v[1]):
            return 1
        else:
            return 0
    def draw(self):
        """
        fonction qui dessine le graphe des dépendances des attributs selon le modèle naive bayes
        Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            attr:le parent des autres attributs
        """
        return drawNaiveBayes(self.dico_proba_a_sachant_t,"target")
    
class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur qui implémente un algorithme estimant la classe d'un individu en utilisant le maximum
    à postériori et l'hypothèse de Naive Bayes, optimisé avec les tests d'indépendance au seuil de x%
    """
    def __init__(self,df,x):
        """
        :param df:le dataframe à tester
        """
        self.df=df
        self.dico_proba_a_sachant_t=dict()
        self.p_t_1=getPrior(self.df)['estimation']#proba t=1
        self.p_t_0=1-self.p_t_1#proba t=0
        for a in utils.getNthDict(df,0):
            if(not isIndepFromTarget(self.df,a,x)):#s"assure que l'atribut n'est pas independant avant de l'ajouter 
                self.dico_proba_a_sachant_t[a]=P2D_l(df,a)
        self.liste_ags_reduits=[i for i in self.dico_proba_a_sachant_t.keys()] #creer la liste des agruments non independants
        
    
    def estimProbas(self,dico_individu):
        """ calcule l'estimation à postriori en utilisant l hypothese du Naive Bayes.
        :param:dico_individu: dictionnaire contenant les informations d'un individu
        :return:un dictionnaire associant à chacune des classes 0 et 1 l'estimation à postériori """
        #def estimProbas(self,dico_individu):
        dico_v={0:1,1:1}
        for a,val_a in dico_individu.items():#pour avoir les noms des attributs
            if a in self.liste_ags_reduits:
                if(a!="target"):
                    tmp=self.dico_proba_a_sachant_t[a]
                    if val_a in tmp[1]:
                        dico_v[1]=dico_v[1]*tmp[1][val_a]
                    else:
                        dico_v[1]=dico_v[1]*0
                    if val_a in tmp[0]:
                        dico_v[0]=dico_v[0]*tmp[0][val_a]
                    else:
                        dico_v[0]=dico_v[0]*0
                   
        dico_v[0]=dico_v[0]*self.p_t_0
        dico_v[1]=dico_v[1]*self.p_t_1
        p_observation=dico_v[0]+dico_v[1]
        if(p_observation!=0):
            dico_v[0]=dico_v[0]/p_observation
            dico_v[1]=dico_v[1]/p_observation

        return dico_v
    
    def estimClass(self,dico_individu):
        """"
        retourne la valeur de la classe ayant le maximum à postriori
        :param: dico_individu:dictionnaire contenant tous les attributs d'un individu
        """ 
        dico_v=self.estimProbas(dico_individu)
        if(dico_v[0]<dico_v[1]):
            return 1
        else:
            return 0
    def draw(self):
        """
        fonction qui dessine le graphe des dépendances des attributs selon le modèle naive bayes
        Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            attr:le parent des autres attributs
        """
        return drawNaiveBayes(self.dico_proba_a_sachant_t,"target")

#------------------------------------------Q7----------------------------------#
def mapClassifiers(dic,df):
    """représente graphiquement les points (préscision,rappel) pour chacun des classifieurs 
    Arguments:
            dic: dictionnaire ayant comme clé nom du classifieur et valeur instance de classfieur 
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
    """
    dico_stat=dict()
    for i,cl in dic.items():
        dico_stat=cl.statsOnDF(df)#on recupere le dictionnaire calculant la precision et le rappel
        pyplot.plot(dico_stat["Précision"],dico_stat["Rappel"],marker="x",color="red")
        pyplot.text(dico_stat["Précision"],dico_stat["Rappel"],str(i))     
    pyplot.show()
    return
#-----------------------------------------Q8-----------------------------------#
def MutualInformation(df ,x, y) :
    """
        Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        x: attribut de df
        y : attribut de df
        Cette fonction rend les informations mutuelles entre x et y
    """
    XY = np.array(df[[x,y]])
    X=[XY[i][0] for i in range(len(XY[:]))]
    Y=[XY[i][1] for i in range(len(XY[:]))]
    vals_uniqueXY, Px_y = np.unique(XY, axis=0, return_counts=True)#vals_uniqueX : ce sont les valeurs uniques des attribus de  X 
    Px_y = Px_y/len(df)
    vals_uniqueX, Px = np.unique(X, return_counts=True)
    Px = Px/len(df)
    vals_uniqueY, Py = np.unique(Y, return_counts=True)
    Py = Py/len(df)
    I = 0
    listeXY=[[a,b] for a,b in vals_uniqueXY]#liste des [xi,yi]
    listeX=[i for i in vals_uniqueX]#liste des xi
    listeY=[i for i in vals_uniqueY]#liste des yi
    for [xi, yi] in listeXY:
        px_y = Px_y[listeXY.index([xi,yi])]#recupere la proba de xi 
        px = Px[listeX.index(xi)]#recupere la proba de xi 
        py = Py[listeY.index(yi)]#recupere la proba de yi 
        if px != 0 and py != 0 and px_y != 0:
            I += px_y * np.log2((px_y)/(px*py))
    return I