[README.md](https://github.com/user-attachments/files/30126617/README.md)
# Quantitative Pattern-Based Framework for Universal Musicological Analysis

**Un framework computationnel intégré pour l'analyse musicologique quantitative, avec attribution historique et culturelle explicite.**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17515815-blue)](https://doi.org/10.5281/zenodo.17515815)
[![License: MIT with Mandatory Attribution](https://img.shields.io/badge/license-MIT%20%2B%20Attribution-lightgrey)](#licence)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#pr%C3%A9requis)

Auteur : **Benseddik Ahmed** · Version 4.0 (Integrated)

---

## Vue d'ensemble

Ce framework propose une chaîne d'analyse computationnelle complète pour la musicologie quantitative, en combinant génération de corpus synthétiques, extraction de descripteurs audio (MIR), analyse statistique comparative, apprentissage non supervisé et supervisé, analyse harmonique/contrapuntique symbolique, et un système original de **détection d'influences musicales inter-culturelles**.

Il a été conçu pour s'appliquer aussi bien à des corpus occidentaux (répertoire baroque, tonal) qu'à des traditions non occidentales (gamelan, clave afro-cubaine, maqâm), avec un souci constant de ne pas plaquer des catégories analytiques occidentales sur des musiques qui obéissent à d'autres logiques.

### Cadre éthique

Le framework part du principe que toute musique émerge d'échanges historiques complexes — colonialisme, migration, résistance, synthèse créative. Il vise à **restaurer cette complexité historique** plutôt qu'à l'effacer derrière des catégories analytiques universalisantes. Quatre principes structurent cette approche :

- **Attribution historique** — créditer les innovations à leurs origines culturelles plutôt que de les neutraliser statistiquement
- **Humilité épistémique** — les résultats sont traités comme des hypothèses computationnelles, non comme des vérités définitives
- **Impératif collaboratif** — l'analyse de musiques non occidentales appelle une collaboration avec des experts culturels
- **Méthodologie décoloniale** — les rapports de pouvoir dans les échanges musicaux sont explicitement pris en compte
- **Science ouverte** — méthodes transparentes, résultats reproductibles, ouverture à la contribution communautaire

---

## Architecture : les 9 modules

| # | Module | Fonction |
|---|--------|----------|
| 1 | Génération de corpus synthétique | Génère un corpus baroque synthétique paramétrable (fréquence d'insertion de motifs BACH, arpèges, etc.) pour valider les modules suivants dans un cadre contrôlé |
| 2 | Analyse statistique de motifs | Détection et quantification de motifs mélodiques/rythmiques dans le corpus |
| 3 | Extraction de descripteurs audio | Extraction de descripteurs MIR (MFCC et autres) via `librosa` sur des fichiers audio réels |
| 4 | Analyse statistique comparative | Comparaison inter-groupes des descripteurs extraits (tests statistiques) |
| 5 | Clustering avancé | K-means, DBSCAN et HDBSCAN (optionnel), avec métriques de qualité (silhouette, Davies-Bouldin, Calinski-Harabasz) |
| 6 | Classification supervisée | Classification par forêt aléatoire (Random Forest) sur les descripteurs extraits |
| 7 | Visualisation dimensionnelle | Projection PCA des espaces de descripteurs |
| 8 | Analyse harmonique et contrapuntique | Analyse symbolique (via `music21`) de partitions MusicXML, avec attribution historique des pratiques harmoniques identifiées |
| 9 | Détection universelle d'influences | Registre d'influences musicales (`UniversalInfluenceRegistry`) et détecteur (`UniversalInfluenceDetector`) qui met en correspondance les résultats du Module 8 avec des règles de détection d'influences historiques, puis visualise le réseau d'influences (`InfluenceNetworkVisualizer`) |

L'orchestration de l'ensemble est assurée par la classe `UniversalMusicologicalFramework`, qui expose une interface unifiée pour lancer tout ou partie du pipeline.

---

## Installation

### Prérequis

- Python ≥ 3.9

### Dépendances

```bash
pip install numpy>=1.21.0 scipy>=1.7.0 pandas>=1.3.0 librosa>=0.9.0 \
            scikit-learn>=1.0.0 matplotlib>=3.4.0 music21>=8.0.0 \
            networkx>=2.6.0
```

Dépendance optionnelle (clustering HDBSCAN) :

```bash
pip install hdbscan>=0.8.27
```

> Le framework détecte automatiquement l'absence de `music21` ou `hdbscan` et désactive proprement les modules correspondants (8 et clustering HDBSCAN) plutôt que d'échouer.

### Récupérer le framework

```bash
git clone https://github.com/benseddikahmed-sudo/Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis.git
cd Quantitative-Pattern-Based-Framework-for-Quantitative-Musicological-Analysis
```

---

## Utilisation rapide

```python
import qpb_music_framework as fw

framework = fw.UniversalMusicologicalFramework()

# Modules 1-2 : corpus synthétique + analyse de motifs
framework.run_synthetic_corpus_analysis(
    corpus_length=1000,
    bach_frequency=0.03,
    arpeggio_frequency=0.05
)

# Pipeline complet (modules activables indépendamment)
results = framework.run_complete_pipeline(
    run_synthetic=True,
    run_audio=False,          # nécessite des répertoires de corpus audio
    run_symbolic=False,       # nécessite un fichier MusicXML
    run_influence=False       # nécessite l'analyse symbolique préalable
)

framework.export_complete_results()
```

Une démonstration minimale est aussi disponible directement dans le script :

```bash
python qpb_music_framework.py
```

### Analyse symbolique et détection d'influences (Modules 8-9)

Le dépôt inclut `test_module8_9_demo.py`, qui illustre l'usage des modules 8 et 9 sur un choral de Bach (BWV 66.6) fourni par le corpus intégré de `music21` :

```bash
python test_module8_9_demo.py
```

Ce script :
1. charge une partition d'exemple depuis `music21.corpus`,
2. l'exporte en MusicXML (le Module 8 attend un chemin de fichier, pas un objet `Stream` en mémoire),
3. lance l'analyse harmonique/contrapuntique (Module 8),
4. lance la détection d'influences (Module 9) sur les résultats obtenus.

Pour l'utiliser sur vos propres partitions :

```python
framework.run_symbolic_analysis('chemin/vers/partition.xml')
framework.run_influence_detection()
```

### Corpus synthétiques inclus

Le dépôt fournit quatre corpus XML synthétiques permettant de tester le framework sur des idiomes contrastés sans dépendre de partitions externes :

- `synthetic_gamelan.xml`
- `synthetic_clave.xml`
- `synthetic_blues.xml`
- `synthetic_maqam.xml`

---

## Structure du dépôt

```
.
├── qpb_music_framework.py     # Framework complet (9 modules)
├── test_module8_9_demo.py     # Démonstration Modules 8-9 (analyse harmonique + influences)
├── synthetic_gamelan.xml      # Corpus synthétique — gamelan
├── synthetic_clave.xml        # Corpus synthétique — clave afro-cubaine
├── synthetic_blues.xml        # Corpus synthétique — blues
└── synthetic_maqam.xml        # Corpus synthétique — maqâm
```

---

## Citation

Si vous utilisez ce framework dans un travail académique, merci de citer :

> Benseddik, A. (2025). *Quantitative Pattern-Based Framework for Universal Musicological Analysis: An Ethical Approach to Computational Music Analysis with Historical Attribution.* GitHub repository. https://github.com/benseddikahmed-sudo/Universal-Musicological-Analysis-Framework

DOI : [10.5281/zenodo.17515815](https://doi.org/10.5281/zenodo.17515815)

## Références académiques mobilisées

- Floyd, S. A. (1995). *The Power of Black Music.* Oxford University Press.
- Bohlman, P. V. (2002). *World Music: A Very Short Introduction.* Oxford University Press.
- Born, G., & Hesmondhalgh, D. (2000). *Western Music and Its Others.* University of California Press.
- Nettl, B. (2015). *The Study of Ethnomusicology* (3e éd.). University of Illinois Press.

## Licence

MIT avec attribution obligatoire — toute réutilisation doit créditer l'auteur et le dépôt d'origine.

## Contribution

Les contributions sont bienvenues, en particulier de la part de spécialistes de traditions musicales non occidentales, conformément à l'« impératif collaboratif » qui structure ce projet. Merci d'ouvrir une *issue* pour discuter d'une évolution avant de soumettre une *pull request* substantielle.
