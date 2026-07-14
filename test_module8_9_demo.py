#!/usr/bin/env python3
"""
Demo: Test Module 8 (Harmonic Analysis) + Module 9 (Influence Detection)
=========================================================================

Utilise une partition intégrée à music21 (choral de Bach, BWV 66.6, à 4 voix)
pour ne pas avoir à chercher un fichier externe. music21 est installé avec
son propre "corpus" d'exemples -- on en prend un ici.

Usage:
    python test_module8_9_demo.py

Prérequis:
    - qpb_music_framework.py doit être dans le même dossier (ou dans le
      PYTHONPATH)
    - music21 doit être installé (pip install music21)
"""

from pathlib import Path
from music21 import corpus

# Import du framework (le fichier qpb_music_framework.py doit être
# dans le même dossier que ce script)
import qpb_music_framework as fw


def main():
    print("=" * 80)
    print("DEMO: Module 8 (Harmonic Analysis) + Module 9 (Influence Detection)")
    print("=" * 80)

    # --------------------------------------------------------------
    # ÉTAPE 1 : Récupérer une partition d'exemple depuis music21
    # --------------------------------------------------------------
    print("\n[1/4] Chargement d'une partition d'exemple (Bach, BWV 66.6)...")

    score = corpus.parse('bach/bwv66.6')

    # Le Module 8 attend un CHEMIN DE FICHIER (pas un objet Stream en
    # mémoire), donc on l'exporte temporairement en MusicXML.
    output_dir = Path("./framework_outputs")
    output_dir.mkdir(exist_ok=True)
    score_path = output_dir / "bach_bwv66_6.xml"
    score.write('musicxml', fp=str(score_path))

    print(f"✓ Partition sauvegardée : {score_path}")
    print(f"  ({len(score.parts)} voix -- choral SATB typique de Bach)")

    # --------------------------------------------------------------
    # ÉTAPE 2 : Lancer le framework complet
    # --------------------------------------------------------------
    print("\n[2/4] Initialisation du framework...")
    framework = fw.UniversalMusicologicalFramework()

    # --------------------------------------------------------------
    # ÉTAPE 3 : Module 8 -- Analyse harmonique et contrapuntique
    # --------------------------------------------------------------
    print("\n[3/4] Lancement du Module 8 (analyse harmonique)...")
    symbolic_results = framework.run_symbolic_analysis(
        score_path=str(score_path),
        acknowledge_influences=True
    )

    print("\n--- Résumé Module 8 ---")
    print(f"Tonalité détectée   : {symbolic_results.get('key')}")
    print(f"Événements harmoniques : {len(symbolic_results['harmony'])}")
    print(f"Cadences détectées  : {len(symbolic_results['cadences'])}")
    print(f"Blue notes          : {symbolic_results['blue_notes']['total_occurrences']}")
    print(f"Ratio syncopation   : {symbolic_results['syncopation_ratio']:.1%}")
    print(f"Texture             : {symbolic_results['texture']}")
    print(f"Quintes/octaves parallèles signalées : {len(symbolic_results['parallel_motion'])}")

    if symbolic_results['cadences']:
        print("\nDétail des cadences :")
        for cad in symbolic_results['cadences'][:10]:
            print(f"  Mesure {cad['measure']:>3} | {cad['type']:>4} | {cad['description']}")

    # --------------------------------------------------------------
    # ÉTAPE 4 : Module 9 -- Détection d'influences historiques
    # --------------------------------------------------------------
    print("\n[4/4] Lancement du Module 9 (détection d'influences)...")
    influence_results = framework.run_influence_detection(
        score_name="Bach - BWV 66.6 (Chorale)"
    )

    print(influence_results['report'])

    # --------------------------------------------------------------
    # Interprétation attendue
    # --------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INTERPRÉTATION ATTENDUE")
    print("=" * 80)
    print("""
Ce choral de Bach est une pièce du répertoire classique européen du
XVIIIe siècle, écrite AVANT les innovations afro-diasporiques du blues
et du jazz (XIXe-XXe siècle). On s'attend donc à :

  - PEU ou PAS de blue notes (tonalité fonctionnelle stricte)
  - PEU de syncopation marquée (choral = rythmique régulière)
  - Cadences PAC/HC très nombreuses (harmonie fonctionnelle classique)
  - Peu ou pas d'influence afro-diasporique/latine détectée

Si le Module 9 ne détecte AUCUNE influence marquante ici, c'est en fait
un BON signe : cela confirme que le détecteur ne génère pas de faux
positifs sur un exemple qui n'a historiquement aucun rapport avec le
blues, le jazz ou les musiques latino-caribéennes.
""")


if __name__ == "__main__":
    main()
