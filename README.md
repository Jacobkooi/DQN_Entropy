# Analyzing Latent Entropy in Deep Q-Learning

This repository contains the research work on **"Analyzing Latent Entropy in Deep Q-Learning"**, as presented in the SSCI 2025 publication.

## Repository Contents

- **Code:** Implementation of the entropy-enhancing methods using the Deep Q-Network architecture.
- **Experiments:** Scripts and configurations for reproducing the maze experiments.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/Jacobkooi/DQN_Entropy.git
   cd DQN_Entropy
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run experiments:

   ```bash
   python main_reward_finding_basic.py   ## Baseline
   python main_reward_finding_basic.py --batch_entropy_scaler=1 --subsequent=1 ## Pair-wise distance maximization
   python main_reward_finding_basic.py --activation='layernorm' ## Layer normalization
   python main_reward_finding_basic.py --activation='layernorm' --batch_entropy_scaler=1 --subsequent=1 ## Layer normalization + Pair-wise distance maximization
   python main_reward_finding_basic.py --activation='tanh+initialization' ## Tanh activation + Xavier initialization (gain 5)
   ```
   
Note that running these experiments automatically creates the representation visualization in a newly created \images directory.

## Research Paper

For detailed methodology and results, please refer to our paper.

## Authors

- **Jacob E. Kooi** - Vrije Universiteit Amsterdam  
  [j.e.kooi@vu.nl](mailto:j.e.kooi@vu.nl)
- **Mark Hoogendoorn** - Vrije Universiteit Amsterdam  
  [m.hoogendoorn@vu.nl](mailto:m.hoogendoorn@vu.nl)
- **Vincent François-Lavet** - Vrije Universiteit Amsterdam  
  [vincent.francoislavet@vu.nl](mailto:vincent.francoislavet@vu.nl)

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{kooi2025latententropy,
  title={Analyzing Latent Entropy in Deep Q-Learning},
  author={Kooi, Jacob E. and Hoogendoorn, Mark and François-Lavet, Vincent},
  booktitle={Symposium Series on Computational Intelligence},
  year={2025}
}
```

## Acknowledgements

This research was conducted at Vrije Universiteit Amsterdam.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
```