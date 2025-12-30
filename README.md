# Universal Musicological Analysis Framework v4.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[DOI]![DOI](https://zenodo.org/badge/DOI/10.5281(https://zenodo.org/badge/DOI/10.5281/zenodo.17515815.svg)](https://doi.org/10.5281/zenodo.17515815)
> *A comprehensive computational musicology framework integrating pattern analysis, clustering, symbolic music processing, and ethical historical attribution.*

**Author:** Benseddik Ahmed  
**Institution:** Independent Digital Humanities Researcher, France  
**Contact:** [benseddik.ahmed@gmail.com]  
**Repository:** https://github.com/benseddikahmed-sudo/Universal-Musicological-Analysis-Framework

---

## 🌟 Key Features

### 🎵 **Multi-Paradigm Analysis**
- **Audio Analysis**: MIR feature extraction (MFCCs, spectral features, temporal descriptors)
- **Symbolic Analysis**: Harmonic analysis, cadence detection, voice leading, contrapuntal rules
- **Pattern Detection**: Statistical validation with binomial tests, motif recognition
- **Clustering**: K-means, DBSCAN, HDBSCAN with automatic hyperparameter optimization

### 🌍 **Ethical Historical Attribution**
- **70+ Documented Influences**: Spanning African diaspora, Arabic/Andalusian, Indian, Asian, Latin American, and more
- **Automatic Detection**: Hypothesizes influences based on musical features
- **Academic Rigor**: Every influence citation-backed with peer-reviewed sources
- **Power Dynamics**: Explicitly recognizes colonialism, appropriation, and resistance

### 📊 **Comprehensive Toolkit**
- Synthetic corpus generation for method validation
- Advanced clustering with density-based algorithms
- PCA visualization for dimensionality reduction
- Random Forest classification for style recognition
- Network visualization of historical influence flows

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Modules](#framework-modules)
- [Usage Examples](#usage-examples)
- [Ethical Framework](#ethical-framework)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Basic Installation
````bash
# Clone the repository
git clone https://github.com/benseddikahmed-sudo/Universal-Musicological-Analysis-Framework.git
cd Universal-Musicological-Analysis-Framework

# Install dependencies
pip install -r requirements.txt
````

### Full Installation (with optional dependencies)
````bash
# Core dependencies
pip install numpy scipy pandas matplotlib scikit-learn librosa

# For symbolic analysis (Module 8)
pip install music21

# For advanced clustering (Module 5)
pip install hdbscan

# For network visualization (Module 9)
pip install networkx

# For notebook examples
pip install jupyter notebook
````

### Dependencies List

**Core:**
- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- librosa >= 0.9.0

**Optional:**
- music21 >= 8.0.0 (symbolic analysis)
- hdbscan >= 0.8.27 (hierarchical clustering)
- networkx >= 2.6.0 (influence networks)

---

## 🚀 Quick Start

### 1. Basic Audio Corpus Analysis
````python
from framework import UniversalMusicologicalFramework

# Initialize framework
framework = UniversalMusicologicalFramework()

# Analyze audio corpus
results = framework.run_audio_corpus_analysis(
    corpus_directories={
        'classical': './corpus/classical',
        'jazz': './corpus/jazz'
    },
    compare_features=['mfcc_mean_1', 'spectral_centroid_mean'],
    run_clustering=True,
    run_classification=True
)

# Results automatically saved to ./framework_outputs/
````

### 2. Symbolic Music Analysis with Historical Attribution
````python
# Analyze a score (e.g., Gershwin's Rhapsody in Blue)
symbolic_results = framework.run_symbolic_analysis(
    score_path='./scores/gershwin_rhapsody_in_blue.xml',
    acknowledge_influences=True
)

# Detect historical influences
influence_results = framework.run_influence_detection(
    score_name="Rhapsody in Blue"
)

# Output:
# ✓ African-diasporic influence detected (85% confidence)
#   • Blue notes: 47 occurrences
#   • Syncopation ratio: 68%
#   • Attribution: Blues/Jazz (W.C. Handy, Louis Armstrong)
#   • Sources: Floyd (1995), Southern (1997)
````

### 3. Complete Integrated Pipeline
````python
# Run all modules
results = framework.run_complete_pipeline(
    run_synthetic=True,      # Modules 1-2: Pattern validation
    run_audio=True,          # Modules 3-7: Audio analysis
    run_symbolic=True,       # Module 8: Harmonic analysis
    score_path='./my_score.xml',
    run_influence=True       # Module 9: Historical attribution
)

# Export all results
framework.export_complete_results()
````

---

## 📦 Framework Modules

### **Module 1-2: Synthetic Corpus & Pattern Analysis**
- Generate controlled pitch-class sequences
- Intentionally embed motifs for validation
- Statistical significance testing (binomial tests)
- Validate pattern detection algorithms

### **Module 3-4: Audio Feature Extraction & Statistics**
- Extract MFCCs, spectral centroids, zero-crossing rates
- Welch's t-tests for group comparisons
- Comprehensive MIR descriptor computation
- CSV export for external analysis

### **Module 5: Advanced Clustering**
- **K-means**: Baseline centroid-based clustering
- **DBSCAN**: Density-based with automatic ε optimization
- **HDBSCAN**: Hierarchical density clustering with noise detection
- Automatic hyperparameter tuning via k-distance graphs
- Quality metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz

### **Module 6: Supervised Classification**
- Random Forest ensemble classifier
- Train/test split with stratification
- Precision, recall, F1-score reporting
- Feature importance analysis

### **Module 7: Dimensional Visualization**
- PCA projection to 2D
- Color-coded by labels or clusters
- High-resolution export (300 DPI)
- Variance explained annotations

### **Module 8: Harmonic & Contrapuntal Analysis**
- **Roman numeral analysis** (functional harmony)
- **Cadence detection** (PAC, IAC, HC, PC, DC)
- **Voice leading analysis** (parallel, contrary, oblique motion)
- **Blue notes detection** (African-American influence)
- **Historical attribution** (automatic recognition)
- Requires: music21, MusicXML/MIDI input

### **Module 9: Universal Influence Recognition**
- **70+ documented influences** with academic sources
- Afro-diasporic (blues, jazz, gospel, R&B)
- Arabic/Andalusian → European (Moorish Spain)
- Indian classical → Western (Beatles, minimalism)
- Gamelan → French impressionism (Debussy)
- Latin/Caribbean → Jazz (Afro-Cuban, bossa nova)
- Network visualization of influence flows
- Power dynamics: colonial, trade, resistance

---

## 📚 Usage Examples

### Example 1: Comparing Baroque vs. Jazz
````python
framework = UniversalMusicologicalFramework()

results = framework.run_audio_corpus_analysis(
    corpus_directories={
        'baroque': './corpus/bach_vivaldi',
        'jazz': './corpus/coltrane_davis'
    },
    compare_features=['zcr_mean', 'spectral_centroid_mean']
)

# Expected Output:
# Comparative Analysis: 'zcr_mean'
# Group 1 (baroque): μ=0.042, σ=0.015
# Group 2 (jazz): μ=0.089, σ=0.034
# t-statistic = -8.23, p-value = 1.2e-08
# Significant difference (α=0.05): Yes
#
# Interpretation: Jazz shows higher zero-crossing rate
# → More "noisy" timbres (saxophones, brass, percussion)
# → Influence of African-diasporic timbral aesthetics
````

### Example 2: Detecting Arabic Influence in Flamenco
````python
symbolic_results = framework.run_symbolic_analysis(
    score_path='./scores/paco_de_lucia_entre_dos_aguas.xml'
)

influence_results = framework.run_influence_detection()

# Expected Detection:
# ✓ ARABIC_ANDALUSIAN influence detected (78% confidence)
#   • Phrygian mode (characteristic of maqam Hijaz)
#   • Melismatic ornamentation
#   • Historical context: Al-Andalus (711-1492 CE)
#   • Key figures: Ziryab, Ibn Bajja
#   • Mechanism: Cultural retention post-Reconquista
#   • Sources: Farmer (1988), Glasser (2016)
````

### Example 3: Beatles' Indian Influence
````python
symbolic_results = framework.run_symbolic_analysis(
    score_path='./scores/beatles_within_you_without_you.xml'
)

influence_results = framework.run_influence_detection(
    score_name="Within You Without You"
)

# Expected Detection:
# ✓ INDIAN_CLASSICAL influence detected (92% confidence)
#   • Drone-based structure (tanpura simulation)
#   • Raga-like modal improvisation
#   • Sitar presence (instrumentation)
#   • Period: 1960-1970 CE
#   • Key figures: Ravi Shankar (teacher), George Harrison
#   • Mechanism: Direct study, counterculture exchange
#   • Sources: Lavezzoli (2006), Farrell (1997)
````

---

## 🌍 Ethical Framework

### Core Principles

1. **Historical Attribution**: Credit innovations to their cultural origins
2. **Epistemic Humility**: Computational hypotheses, not definitive truths
3. **Collaborative Imperative**: Work with cultural experts for non-Western analysis
4. **Decolonial Methodology**: Recognize power dynamics in musical exchange
5. **Open Science**: Transparent methods, reproducible results

### What This Framework Does

✅ **Recognizes debt**: Shows how Western music builds on non-Western innovations  
✅ **Documents influences**: 70+ citations to peer-reviewed academic sources  
✅ **Contextualizes power**: Distinguishes colonialism, appropriation, trade, resistance  
✅ **Remains humble**: "Hypotheses" not "discoveries"  
✅ **Enables dialogue**: Open-source for community contribution

### What This Framework Does NOT Do

❌ **Extract without consent**: No analysis of sacred/restricted musics  
❌ **Claim universality**: No "universal music language" rhetoric  
❌ **Impose Western categories**: Respects emic (internal) cultural frameworks  
❌ **Appropriate**: All influences cited with proper attribution  
❌ **Essentialize**: No "African music is X" generalizations

### Ethical Use Guidelines

**DO:**
- Use for educational purposes highlighting cross-cultural exchange
- Cite cultural origins when discussing detected influences
- Collaborate with tradition-bearers for non-Western analysis
- Acknowledge limitations of computational methods

**DON'T:**
- Analyze sacred musics without community permission
- Present hypotheses as historical facts
- Use detected influences to make essentialist claims
- Ignore the "hypothetical" nature of automated detections

---

## 📖 Documentation

### Full Documentation
- **User Guide**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Theoretical Background**: [docs/THEORETICAL_FOUNDATIONS.md](docs/THEORETICAL_FOUNDATIONS.md)
- **Jupyter Notebooks**: [notebooks/](notebooks/)

### Tutorials
1. [Getting Started](notebooks/01_getting_started.ipynb)
2. [Audio Analysis Deep Dive](notebooks/02_audio_analysis.ipynb)
3. [Symbolic Analysis & Attribution](notebooks/03_symbolic_analysis.ipynb)
4. [Advanced Clustering Techniques](notebooks/04_clustering.ipynb)
5. [Historical Influence Networks](notebooks/05_influence_networks.ipynb)

### Academic Papers
- **Methodology Paper**: [papers/methodology.pdf](papers/methodology.pdf)
- **Case Studies**: [papers/case_studies/](papers/case_studies/)

---

## 📊 Output Structure
````
framework_outputs/
├── results/
│   ├── synthetic_corpus_analysis.json
│   ├── audio_features.csv
│   ├── clustering_comparison.csv
│   ├── symbolic_analysis_summary.json
│   ├── influence_detection.json
│   └── complete_framework_results.json
│
└── visualizations/
    ├── pca_by_label.png
    ├── pca_by_clusters.png
    ├── harmonic_analysis.png
    ├── voice_leading_distribution.png
    └── influence_network.png
````

---

## 📝 Citation

### Software Citation
````bibtex
@software{benseddik2025universal,
  author = {Benseddik, Ahmed},
  title = {Quantitative Pattern-Based Framework for Universal Musicological 
           Analysis: An Ethical Approach to Computational Music Analysis with 
           Historical Attribution},
  year = {2025},
  version = {4.0},
  publisher = {GitHub},
  url = {https://github.com/benseddikahmed-sudo/Universal-Musicological-Analysis-Framework},
  doi = {10.5281/zenodo.XXXXXXX}
}
````

### Paper Citation (when published)
````bibtex
@article{benseddik2025ethical,
  author = {Benseddik, Ahmed},
  title = {An Ethical Computational Framework for Universal Musicological Analysis: 
           Restoring Historical Complexity Through Algorithmic Attribution},
  journal = {Digital Scholarship in the Humanities},
  year = {2025},
  volume = {XX},
  number = {X},
  pages = {XXX--XXX},
  doi = {10.1093/llc/fqXXXXX}
}
````

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

1. **Add Influences**: Document new historical influences with academic sources
2. **Bug Reports**: Open issues for bugs or unexpected behavior
3. **Feature Requests**: Suggest new analytical capabilities
4. **Documentation**: Improve tutorials, examples, translations
5. **Case Studies**: Share your research using this framework

### Adding New Influences
````python
from framework import UniversalInfluenceRegistry, MusicalInfluence

registry = UniversalInfluenceRegistry()

# Add a new documented influence
registry.add_influence(
    MusicalInfluence(
        source_tradition="Your Source Tradition",
        target_tradition="Your Target Tradition",
        period="Historical Period",
        mechanism="How influence occurred",
        musical_elements=["Element 1", "Element 2"],
        key_figures=["Figure 1", "Figure 2"],
        academic_sources=[
            "Author (Year). Title. Publisher.",
            "Author (Year). Title. Journal."
        ],
        certainty='established',  # or 'probable', 'speculative'
        power_dynamics='trade'  # or 'colonial', 'voluntary_exchange', etc.
    )
)
````

**Requirement**: All influences MUST include academic sources.

---

## 📜 License

This project is licensed under the **MIT License with Attribution Requirement**.

**You are free to:**
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Use privately

**Under the conditions:**
- 📝 Include original copyright notice
- 📝 Include license text
- 📝 Cite this framework in academic publications
- 📝 Acknowledge cultural origins when discussing detected influences

See [LICENSE](LICENSE) for full terms.

---

## 🙏 Acknowledgments

### Academic Foundations

This framework builds on the scholarship of:

- **Samuel A. Floyd** - *The Power of Black Music* (African-diasporic influences)
- **Eileen Southern** - *The Music of Black Americans* (historical documentation)
- **Philip V. Bohlman** - *World Music: A Very Short Introduction* (critical musicology)
- **Bruno Nettl** - *The Study of Ethnomusicology* (ethical frameworks)
- **Georgina Born & David Hesmondhalgh** - *Western Music and Its Others* (postcolonial critique)

### Technical Inspiration

- **music21**: Michael Scott Cuthbert et al. - Symbolic music analysis
- **librosa**: Brian McFee et al. - Audio analysis toolkit
- **scikit-learn**: Pedregosa et al. - Machine learning infrastructure

### Cultural Consultants (Placeholder)

*This section will acknowledge cultural experts consulted for non-Western music analysis once collaborations are established.*

---

## 📧 Contact

**Benseddik Ahmed**  
Independent Digital Humanities Researcher  
France

- 📧 Email: [your-email@domain.com]
- 🔗 ORCID: [0009-0005-6308-8171](https://orcid.org/0009-0005-6308-8171)
- 🐦 Twitter: [@YourHandle]
- 💼 LinkedIn: [Your Profile]

---

## 🔗 Related Projects

- [ArcheoPatterns-GPS Framework](https://github.com/benseddikahmed-sudo/ArcheoPatterns-GPS-Framework) - Computational archaeology
- [Ancient Text Numerical Analysis](https://github.com/benseddikahmed-sudo/Ancient-Text-Numerical-Analysis) - Hebrew manuscript analysis

---

## 📅 Changelog

### v4.0 (2025-01-XX) - Initial Integrated Release
- ✨ Complete integration of 9 modules
- ✨ 70+ documented historical influences
- ✨ Automatic hyperparameter optimization
- ✨ Ethical attribution system
- 📚 Comprehensive documentation
- 🎓 Jupyter notebook tutorials

### v3.0 (2024-12-XX) - Module 8 & 9 Development
- Added harmonic analysis with blue notes detection
- Implemented universal influence registry
- Created network visualization system

### v2.0 (2024-11-XX) - Advanced Clustering
- Added DBSCAN and HDBSCAN
- Implemented automatic parameter optimization
- Enhanced evaluation metrics

### v1.0 (2024-10-XX) - Core Framework
- Initial release with Modules 1-7
- Basic audio and pattern analysis

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=benseddikahmed-sudo/Universal-Musicological-Analysis-Framework&type=Date)](https://star-history.com/#benseddikahmed-sudo/Universal-Musicological-Analysis-Framework&Date)

---

**Made with ❤️ for ethical computational musicology**

*"Music is the result of centuries of cross-cultural exchange. This framework honors that complexity."*
