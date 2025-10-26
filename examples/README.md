# Neural Network Examples

Cette section contient des exemples démontrant l'utilisation du réseau de neurones avec MCTS.

## Programmes disponibles

### `demo_nn_mcts`

Programme de démonstration complet montrant:

1. **État encoding**: Comment convertir un `RuleState` en tensor
2. **Network inference**: Forward pass du réseau (policy + value)
3. **MCTS comparison**: Comparaison MCTS avec et sans réseau
4. **Gradient flow**: Test du backward pass

## Compilation

### Avec CMake (recommandé)

```bash
# Depuis la racine du projet
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target demo_nn_mcts

# Lancer la démo
./examples/demo_nn_mcts
```

### Avec le script de build

```bash
# Depuis la racine du projet
bash scripts/build.sh

# Lancer la démo
./build/examples/demo_nn_mcts
```

## Exemple de sortie attendue

```
╔═══════════════════════════════════════════════════════════════════╗
║  GOLOMB NEURAL NETWORK + MCTS DEMONSTRATION                       ║
║  AlphaGo-style architecture implemented from scratch              ║
╚═══════════════════════════════════════════════════════════════════╝

=== DEMO 1: STATE ENCODING ===

Encoder configuration:
  Upper bound: 50
  Target marks: 6
  Encoding size: 206 features
    - Positions: 51 bits
    - Distances: 51 bits
    - Metadata: 4 features

...

╔═══════════════════════════════════════════════════════════════════╗
║  ALL DEMOS COMPLETED SUCCESSFULLY!                                ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Ce que démontrent ces exemples

### Architecture du réseau

Le `GolombNet` implémente une architecture inspirée d'AlphaGo:

```
Input (encoded state) → [206 features]
    ↓
Hidden Layer 1 [64 neurons] + ReLU
    ↓
Hidden Layer 2 [64 neurons] + ReLU
    ↓
    ├─→ Policy Head + Softmax → P(action) [51 outputs]
    └─→ Value Head + Tanh → V(state) [-1, 1]
```

### Fonctionnalités démontrées

1. **Forward pass**: Le réseau peut prédire:
   - Quelle position placer ensuite (policy)
   - La qualité de l'état actuel (value)

2. **Backward pass**: Tous les gradients sont calculés correctement
   - Prêt pour l'entraînement
   - Gradient checking validé dans les tests

3. **MCTS integration**: Le réseau guide la recherche:
   - Policy priors → biais vers positions prometteuses
   - Value estimation → évaluation rapide des feuilles

## Notes

- **Réseau non entraîné**: Les exemples utilisent un réseau avec poids aléatoires
- **Après entraînement**: Le réseau devrait significativement améliorer MCTS
- **Performance**: Version démo avec architecture réduite (64 neurones vs 256 en production)

## Prochaines étapes

Pour entraîner le réseau:

1. Implémenter optimiseur (SGD/Adam)
2. Générer données via self-play
3. Entraîner avec loss AlphaGo: `L = (z - v)² - π^T log(p)`
4. Évaluer le réseau entraîné vs heuristique

Voir `docs/NEURAL_NETWORK.md` pour plus de détails (à venir).
