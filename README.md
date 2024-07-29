# Codes thèses

Les codes réalisés dans le cadre de ma thèse intitulée « Analyse géomorphologique par apprentissage profond et imagerie satellitaire multi-source »

## MorNet

Le dossier MorNet contient le code qui accompagne l'article "Multi-source deep-learning approach for automatic geomorphological mapping: the case of glacial moraines" (https://doi.org/10.1080/10095020.2023.2292587)

**1. Prérequis**
- Python version 3.9.7
- Tensorflow version 2.7.0
- Rasterio version 1.2.10
- Numpy version 1.21.4
- scikit-learn 1.0.1

**2. Contenu du dossier**

Deux-sous-dossiers :
- data : qui contiendra les données d'entrées des modèles dont la vérité terrain (ROI_couche_positive.tif) et données satellitaires (voir dictionnaires dans parameters.py)
- modules : qui contient tous les codes python appelés par le code principal main.py

Deux fichiers :
 - main.py : pour lancer le modèle et changer les moraines attribuées à chaque dataset (train, validation et test)
 - Config.cfg : fichier à remplir pour définir tous les paramètres (nombre de filtre, taux de dropout, type de couche, ...)

**3. Commande lancement Linux**

python main.py Config.cfg

## MoraineClassification

Ce dossier présente le modèle d'apprentissage semi-supervisé basé sur des graphes utilisé dans la thèse pour associer les moraines à leurs âges de formation (M1 à M6) à partir de leurs caractéristiques géomorphologiques. Ce code utilise le modèle de propagation des labels CAMLP développé par Yamaguchi et al. (2016). Afin de trouver la bonne combinaison de moraines étiquetées, une méthode de Monte Carlo permet de simuler des combinaisons choisies aléatoirement. Une étude de la contribution de chaque caractéristique est faite via une méthode de bootstrap et l'ajout de données paléoclimatique aux caractéristiques de base.

**1. Prérequis**
- gbssl : l'implémentation de CAMLP disponible ici : https://github.com/junliangma/gbssl
- numpy
- pandas
- sklearn
- matplotlib
- datetime
- cmcrameri : la palette de couleurs scientifiques disponible ici : https://www.fabiocrameri.ch/colourmaps/

**2. Contenu du dossier**

Deux-sous-dossiers :
- data : qui contiendra les données d'entrées des modèles (voir les sections « VARIABLES » dans les codes)
- results : qui contiendra les données de sorties des modèles.

Deux fichiers Python :
- semi_supervised_classif_main.py : calcule la performance moyenne du modèle sur les 30 meilleures simulations de Monte Carlo
- semi_supervised_classif_bootstrap : réalise le bootstrap et l'ajout des données paléoclimatiques
