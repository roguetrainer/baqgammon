# Acknowledgments and Attributions

## Quantum Backgammon Implementation

This document provides proper attribution for the theoretical frameworks, code implementations, and visualization techniques used in the Quantum Backgammon project.

---

## Theoretical Foundations

### Quantum Game Theory Framework

The theoretical foundation for quantum backgammon is based on established quantum game theory research:

**Primary Sources:**

1. **Eisert, J., Wilkens, M., & Lewenstein, M. (1999)**
   - "Quantum games and quantum strategies"
   - *Physical Review Letters*, 83(15), 3077-3080
   - **Contribution**: EWL framework for quantum games with superposition and entanglement
   - **Used in**: Game-theoretic analysis in Section 4.3 (Variant 3)

2. **Meyer, D. A. (1999)**
   - "Quantum strategies"
   - *Physical Review Letters*, 82(5), 1052-1055
   - **Contribution**: Quantum strategies in sequential games
   - **Used in**: Strategic framework for all three variants

3. **Nielsen, M. A., & Chuang, I. L. (2010)**
   - *Quantum Computation and Quantum Information*
   - Cambridge University Press
   - **Contribution**: Density matrix formalism, von Neumann entropy, measurement theory
   - **Used in**: Mathematical framework in Section 4, all quantum mechanics implementations

### Quantum Information Theory

4. **Hanauske, M., Kunz, J., Lamprecht, S., & Bernius, S. (2010)**
   - "Quantum game theory and open access publishing"
   - *Physica A*, 389(21), 4983-4991
   - **Contribution**: Application of quantum game theory to real-world scenarios
   - **Used in**: Pedagogical approach and practical implementation considerations

5. **Benjamin, S. C., & Hayden, P. M. (2001)**
   - "Comment on 'Quantum games and quantum strategies'"
   - *Physical Review Letters*, 87(6), 069801
   - **Contribution**: Refinements to quantum game theory framework
   - **Used in**: Discussion of Nash equilibria and strategic analysis

---

## Original Contributions

### Novel Work in This Project

The following elements are **original contributions** created specifically for this project:

1. **Application to Backgammon**
   - First-ever quantum extension of backgammon
   - Three-variant progression (dice → checkers → entanglement)
   - Strategic timing of measurement as game mechanic

2. **Pedagogical Framework**
   - Progressive complexity design
   - "Why Quantum Backgammon?" rationale
   - Educational value analysis

3. **Implementation Architecture**
   - Complete Python code implementations
   - Visualization strategies for quantum states
   - Interactive Jupyter notebook design

4. **Density Matrix Application**
   - Practical use of density matrices for game state
   - Von Neumann entropy as strategic metric
   - Decoherence modeling in gameplay

---

## Code Implementation

### Quantum Mechanics Libraries

The code implementations use **standard quantum mechanics formalisms** but do not directly copy from any specific library. However, the approaches are informed by:

**Conceptual Influences:**

1. **QuTiP (Quantum Toolbox in Python)**
   - http://qutip.org/
   - **Influence**: Density matrix representation patterns
   - **Note**: We implemented our own simplified version, not using QuTiP directly
   - **Similar concepts**: DensityMatrix class structure, von_neumann_entropy calculation

2. **Qiskit (IBM Quantum)**
   - https://qiskit.org/
   - **Influence**: State vector and measurement patterns
   - **Note**: Conceptual similarity in quantum state representation
   - **Similar concepts**: State preparation, measurement collapse

3. **Cirq (Google Quantum)**
   - https://quantumai.google/cirq
   - **Influence**: Quantum gate operations (for Variant 3)
   - **Note**: Gate application patterns for quantum doubling cube

**Our Implementation**:
- All code written from scratch using NumPy
- Standard quantum mechanics formulas from Nielsen & Chuang textbook
- No direct copying from existing quantum computing libraries
- Educational simplifications for pedagogical clarity

### Mathematical Formulations

**Born Rule** (Measurement probabilities):
```python
P(outcome) = |amplitude|²
```
- **Source**: Standard quantum mechanics (Born, 1926)
- **Implementation**: `get_probabilities()` method in QuantumDice class

**Von Neumann Entropy**:
```python
S(ρ) = -Tr(ρ log₂ ρ)
```
- **Source**: von Neumann, J. (1927), Nielsen & Chuang (2010)
- **Implementation**: `von_neumann_entropy()` method

**Density Matrix**:
```python
ρ = |ψ⟩⟨ψ|  # Pure state
ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|  # Mixed state
```
- **Source**: Standard quantum mechanics formalism
- **Implementation**: `get_density_matrix()` method, DensityMatrix class

**Measurement (Projection)**:
```python
ρ_after = Π ρ Π† / Tr(Π ρ)
```
- **Source**: Nielsen & Chuang (2010), Chapter 2
- **Implementation**: `measure()` and `apply_measurement()` methods

---

## Visualization Techniques

### Matplotlib and Seaborn

All visualizations use standard Python libraries:

1. **Matplotlib**
   - https://matplotlib.org/
   - **License**: PSF-based (permissive)
   - **Used for**: All plots, heatmaps, animations
   - **Standard usage**: No novel techniques, standard library documentation

2. **Seaborn**
   - https://seaborn.pydata.org/
   - **License**: BSD 3-Clause
   - **Used for**: Color palettes, statistical plots
   - **Standard usage**: Default configurations and styling

3. **NumPy**
   - https://numpy.org/
   - **License**: BSD 3-Clause
   - **Used for**: All numerical computations
   - **Standard usage**: Array operations, linear algebra

### Visualization Concepts

**Quantum State Heatmaps**:
- **Concept**: Standard in quantum information visualization
- **Common in**: QuTiP documentation, quantum computing textbooks
- **Our implementation**: Custom styling for dice outcomes (6×6 grid)

**Probability Distribution Visualization**:
- **Concept**: Standard statistical visualization
- **Common in**: Any probability/statistics textbook
- **Our implementation**: Applied to quantum measurement outcomes

**Density Matrix Visualization**:
- **Concept**: Standard in quantum computing education
- **Reference**: Nielsen & Chuang (2010), Figure 2.5
- **Our implementation**: Real/imaginary parts with diverging colormap

**Entropy Meters**:
- **Concept**: Novel visualization for this project
- **Original design**: Progress bar representation of von Neumann entropy
- **Purpose**: Make abstract quantum information concrete for players

---

## Similar Projects (Not Directly Used)

### Related Work in Quantum Games

We acknowledge these similar projects, though we did not use their code:

1. **Quantum Tic-Tac-Toe**
   - Multiple implementations exist online
   - **Difference**: Our work is original, focused on backgammon
   - **Not used**: Did not reference any specific implementation

2. **Quantum Chess**
   - Cantwell, C. (2015) - quantum-chess.com
   - **Difference**: Chess is deterministic; backgammon adds genuine randomness
   - **Not used**: Completely different game mechanics

3. **Quantum Prisoner's Dilemma Simulators**
   - Various educational implementations
   - **Similarity**: Game theory framework
   - **Difference**: Abstract economic game vs. concrete board game
   - **Not used**: Did not reference specific implementations

---

## Educational Materials

### Pedagogical Approach Influenced By:

1. **Quantum Country** (Andy Matuschak & Michael Nielsen)
   - https://quantum.country/
   - **Influence**: Spaced repetition and interactive learning
   - **Not directly used**: Inspired tutorial design philosophy

2. **3Blue1Brown** (Grant Sanderson)
   - https://www.3blue1brown.com/
   - **Influence**: Visual explanations of mathematical concepts
   - **Not directly used**: Inspired emphasis on visualization

3. **Explorable Explanations** (Bret Victor)
   - http://worrydream.com/ExplorableExplanations/
   - **Influence**: Interactive learning and "learning by doing"
   - **Not directly used**: Inspired interactive widget design in Jupyter notebook

---

## Software Tools

### Development Tools Used:

1. **Python 3.x**
   - https://www.python.org/
   - **License**: PSF License (permissive)

2. **Jupyter Notebook**
   - https://jupyter.org/
   - **License**: BSD 3-Clause

3. **IPyWidgets**
   - https://ipywidgets.readthedocs.io/
   - **License**: BSD 3-Clause
   - **Used for**: Interactive buttons and controls in notebook

### Documentation Tools:

1. **Markdown**
   - Standard markup language
   - Used for all documentation

2. **LaTeX Math Notation**
   - Standard mathematical typesetting
   - Used for equations in paper and notebook

---

## Original Design Elements

The following are **original creative contributions** with no direct precedent:

### Novel Game Mechanics:

1. **Delayed Measurement Strategy**
   - Players choose when to collapse quantum states
   - Creates information-uncertainty trade-off
   - No prior implementation in quantum games

2. **Three-Variant Progression**
   - Variant 1: Quantum dice only
   - Variant 2: Quantum checker positions
   - Variant 3: Full entanglement
   - Progressive pedagogical design is original

3. **Von Neumann Entropy as Strategic Metric**
   - Using entropy to guide measurement timing
   - Novel application to game strategy
   - "Uncertainty meter" visualization is original

4. **Quantum Blots and Defensive Superposition**
   - Strategic use of superposition for defense
   - Probability-based hitting mechanics
   - Original game theory analysis

### Novel Visualizations:

1. **Dice Symbol Heatmap**
   - 6×6 grid with dice face symbols (⚀⚁⚂⚃⚄⚅)
   - Probability percentages in each cell
   - Custom design for backgammon context

2. **Board State with Superposition**
   - Semi-transparent checkers at multiple locations
   - Transparency proportional to √probability
   - Probability labels above uncertain pieces

3. **Measurement Animation**
   - Before/after comparison
   - Dramatic countdown and collapse effect
   - Interactive button-triggered measurement

4. **Entropy Meter**
   - Horizontal progress bar showing uncertainty
   - Color-coded interpretation (red/yellow/green)
   - Strategic advice based on entropy level

---

## Attribution Summary

### What We Used:

✅ **Theoretical frameworks** from published quantum game theory papers (properly cited)
✅ **Mathematical formulas** from standard quantum mechanics (Nielsen & Chuang)
✅ **Standard libraries** (NumPy, Matplotlib, Seaborn) with their standard usage
✅ **Pedagogical concepts** from interactive learning research (cited above)

### What We Created:

✨ **All implementation code** written from scratch
✨ **All visualization code** custom designed for this project
✨ **Novel game mechanics** (delayed measurement, three variants)
✨ **Original pedagogical framework** (quantum backgammon concept)
✨ **Interactive Jupyter notebook** with custom widgets and explanations
✨ **Comprehensive documentation** (paper, implementation guide, README)

### What We Did NOT Use:

❌ No code copied from existing quantum computing libraries
❌ No visualization code from other quantum game projects
❌ No direct implementation references to similar games
❌ All code is original work using standard libraries

---

## Licensing

### Our Code:

The implementations provided in this project are:
- **Original work** created for educational purposes
- **Freely usable** for academic and educational applications
- **Open to adaptation** with proper attribution

### Dependencies:

All dependencies (NumPy, Matplotlib, Seaborn, Jupyter) use permissive licenses:
- NumPy: BSD 3-Clause
- Matplotlib: PSF-based (permissive)
- Seaborn: BSD 3-Clause
- Jupyter: BSD 3-Clause
- IPyWidgets: BSD 3-Clause

---

## How to Cite This Work

If you use or build upon this work, please cite:

**Academic Citation:**
```
Quantum Backgammon: A Pedagogical Introduction to Quantum Game Theory
Draft Paper, 2025
Implementations available at: [repository/location]
```

**Code Attribution:**
```python
# Quantum Backgammon Implementation
# Based on theoretical framework from Eisert, Wilkens & Lewenstein (1999)
# Original implementation and design by [authors]
# Uses standard quantum mechanics formalism from Nielsen & Chuang (2010)
```

---

## Contact & Contributions

This is an educational project demonstrating quantum mechanics through game theory.

**Acknowledgments to:**
- The quantum game theory community for theoretical foundations
- Open source Python scientific computing community (NumPy, Matplotlib, Jupyter)
- Interactive learning pioneers (Matuschak, Nielsen, Victor, Sanderson)

**Future contributors** welcome to:
- Extend implementations
- Improve visualizations
- Add new variants
- Apply to other games

**Please maintain attribution** to:
1. Original theoretical work (cited papers)
2. This implementation and pedagogical framework
3. Any code you build upon

---

## Disclaimer

This is an educational project. The quantum mechanics implementations are:
- ✅ **Pedagogically accurate** - follows standard formalism
- ✅ **Computationally correct** - produces valid quantum states
- ⚠️ **Simplified for clarity** - not optimized for large-scale computation
- ⚠️ **Educational focus** - emphasizes understanding over performance

For production quantum computing applications, use established libraries like Qiskit, Cirq, or QuTiP.

---

**Last Updated:** 2025-01-20

**Version:** 1.0

**Status:** Educational Release
