# Prédiction conforme séquentielle pour les problèmes d'images

En intelligence artificielle, plus précisément en machine learning, la prédiction conforme est une méthode de classification qui consiste à construire, pour une image donnée, non pas une unique prédiction mais un sous-ensemble $\mathcal{C}$ de l'ensemble des classes pour un niveau de confiance $\alpha\in ]0;1[$. C’est-à-dire que si $Y$ est une variable aléatoire désignant le vrai label d’une observation qu’on tire aléatoirement, alors l’ensemble $\mathcal{C}$ vérifie :
```math
\mathbb{P}(Y \in \mathcal{C}) \geq \alpha.
```

Contrairement à la classification classique où nous n’avons qu’une seule prédiction, la prédiction conforme en donne plusieurs en sortie avec un score de conformité (la probabilité que celle-ci soit correcte) pour donner à l’utilisateur/trice le privilège de choisir parmi elles en optant pour le plus haut score ou bien pour une autre option qu’il/elle aura choisi en s’aidant de caractéristiques plus fines, en cas de doute entre plusieurs options. Les scores de conformité indiquent également le degré de certitude quant à la prédiction proposée, ce qui donne une confiance au procédé en question et au résultat final.

On pose $`\mathcal{Z} = \{(x,y)\mid x\in\mathcal{X} \text{ et } y\in\mathcal{Y} \text{ est associé à } x \}`$ l'espace de données. On définit la suite de variables aléatoires **Z** $`= (Z_1,\dots,Z_n)`$ avec $Z_i = (X_i,Y_i)$ où $Y_i\in\mathcal{Y}$ est le label correspondant à la donnée $X_i\in\mathcal{X}$. On les suppose toutes indépendantes et identiquement distribuées selon une distribution de probabilité. On pose $z = (z_1,\dots,z_n)$ une réalisation de **Z**, qu'on prendra comme données d'entraînement. Soit $(X,Y)\in\mathcal{Z}$ une nouvelle donnée choisie aléatoirement sur $\mathcal{Z}$.

On dispose de $K$ agents. Pour $\alpha\in]0;1[$, avec un niveau de confiance $1-\alpha$, on définit $\mathcal{C}_k=\mathcal{C}_k(X,z)$ comme l'ensemble de prédiction donné par l'agent $k$, étant donné les données d'entraînement $z$. Alors, on a : 
```math
\forall k\in\{1,\dots,K\},\, \mathbb{P}(Y\in \mathcal{C}_k) \geq 1-\alpha$.   (1)
```


Le label $Y$ étant choisi aléatoirement, il peut bien évidemment être une variable aléatoire. De ce fait, la distribution de la probabilité utilisée dans la propriété $(1)$ est celle de $($**Z**$,Y)$. On dit que $\mathcal{C}_k$ a une couverture *exacte* si $\mathbb{P}(Y\in\mathcal{C}_k)=1-\alpha$.

Il faut maintenant agréger tous ces ensembles pour créer un nouvel ensemble de sorte à être optimal en couverture et en taille. Le but est alors de trouver *la* méthode d'agrégation qui approche répond aux critères d'erreur et de taille, le tout avec la plus petite valeur de $K$ possible.

Veuillez trouver l'ensemble de mon travail dans le notebook Python intitulé "PFE Master Maths Info - Manorathan JEEVAKAN.ipynb". Pour plus de détails sur le projet sur l'aspect théorique, vous pouvez les trouver dans mon rapport de stage joint à ce dépôt.

## Sources

V. Vovk, A. Gammerman, and G. Shafer. *Algorithmic Learning in a Random World*, volume 29. Springer, 2005. (https://link.springer.com/book/10.1007/b106715)

Matteo Gasparin, Aaditya Ramdas *Merging uncertainty sets via majority vote*, March 2024 (https://arxiv.org/html/2401.09379v4)
