# AGENTS.MD

## 1\. The Core Mission

Your primary objective is to refactor the existing proof-of-concept repository into a robust, flexible, and extensible research framework. This involves replacing the current collection of single-purpose scripts with a single, powerful, configuration-driven training script. The new framework will replace the simple MLP optimizer with a more powerful `PerceiverIO` model and adopt modern standards like `safetensors` for all weight handling.

-----

## 2\. Guiding Principles

You must adhere to the following architectural principles throughout the implementation:

  * **Configuration Over Hardcoding**: All experimental parameters (model names, learning rates, epochs, file paths, etc.) must be configurable via command-line arguments or a central configuration file. There should be no hardcoded values in the core logic.
  * **Modularity and Separation of Concerns**: Each component (data loading, model definitions, utility functions, training logic) must be in its own dedicated file. This ensures the codebase is easy to read, maintain, and extend.
  * **Architecture Agnosticism**: The core training script must not be tied to any single model architecture. It should be possible to train a new target model (e.g., a ResNet) or a new optimizer model by simply changing a configuration value, without altering the training script itself.
  * **Safety and Modern Standards**: All model weights and checkpoints must be saved and loaded using the `safetensors` format for safety, speed, and interoperability.

-----

## 3\. Project Architecture

You will refactor the existing files into the following directory structure:

```
/
├── train.py                  # The new, centralized master training script.
├── utils.py                  # Helper functions (safetensors, model factory).
├── dataset.py                # The enhanced WeightTrajectoryDataset.
├── models/
│   ├── __init__.py
│   ├── target_cnn.py         # The existing CNN model.
│   └── perceiver_optimizer.py  # The new PerceiverIO-based optimizer.
├── data/                       # (for MNIST dataset)
├── trajectories/               # Directory for storing saved weight trajectories.
└── checkpoints/                # Directory for saving optimizer model checkpoints.
```

-----

## 4\. Detailed Implementation Tasks

### Task 4.1: Implement `utils.py`

Create a file named `utils.py` containing the following helper functions:

  * A function `get_target_model(name: str)` that takes a model name as a string and returns an initialized instance of that model (e.g., `TargetCNN`).
  * A function `get_optimizer_model(name: str)` that returns an instance of a generative optimizer model (e.g., `PerceiverOptimizer`).
  * A function `save_model_weights(model, filepath)` that saves a model's state dictionary to the specified path using `safetensors.torch.save_file`.
  * A function `load_model_weights(model, filepath)` that loads weights from a `.safetensors` file into a model instance.

### Task 4.2: Implement `dataset.py`

Create a file named `dataset.py` containing the `WeightTrajectoryDataset` class. This class must:

  * Be initialized with a `trajectory_dir` and a `mode` argument ('sequential' or 'permutation').
  * In 'sequential' mode, it should provide consecutive pairs of weights `(W_t, W_{t+1})`.
  * In 'permutation' mode, it should provide all possible forward-moving pairs of weights `(W_i, W_j)` where `i < j`.
  * The `__getitem__` method should return the start weights, the end weights, and their corresponding timesteps `t_start` and `t_end`.

### Task 4.3: Implement `models/perceiver_optimizer.py`

Create the file for the `PerceiverOptimizer` class within the `models/` directory. This class must:

  * Be a `torch.nn.Module`.
  * Use the `PerceiverIO` model from the repository as its core component.
  * Be initialized with hyperparameters for the Perceiver (e.g., depth, num\_latents, latent\_dim).
  * Its `forward` method must accept a flattened weight tensor and a timestep tensor. For the permutation training, it must accept a start weight tensor, a start time, and an end time.
  * The implementation must correctly reshape and combine the weight and time inputs into a single sequence for the Perceiver, and reshape the output back into a flattened weight tensor. This implementation must be architecture-agnostic.

### Task 4.4: Implement `train.py`

This will be the main entry point for all operations. It must:

  * Use Python's `argparse` library to handle command-line arguments.
  * Have two main modes of operation (phases): `train-target` and `train-optimizer`.
  * **In `train-target` mode:**
      * Dynamically load the target model architecture specified by a `--target-model` flag.
      * Train it on a dataset (e.g., MNIST).
      * Allow the optimizer (e.g., 'Adam', 'SGD') and learning rate to be set via flags.
      * Save the weight trajectory at each epoch to a uniquely named directory (specified by a `--run-name` flag) as `.safetensors` files.
  * **In `train-optimizer` mode:**
      * Dynamically load the generative optimizer model (e.g., `PerceiverOptimizer`) specified by a `--optimizer-model` flag.
      * Load the appropriate weight trajectory based on the `--run-name` flag.
      * Use the `WeightTrajectoryDataset`, respecting the `--training-mode` flag ('sequential' or 'permutation').
      * Implement a standard training loop to train the generative optimizer.
      * Implement checkpointing: save the model state periodically to the `checkpoints/` directory.
      * Implement resuming: if a `--resume-from` flag is provided with a path to a checkpoint, the script must load the model state and continue training from that point.

-----

## 5\. Testing & Experimentation Protocol

After implementing the above, you are to verify your work by running the following experimental protocol:

1.  **Execute Phase 1:** Run `train.py train-target` for the `TargetCNN` model for 10 epochs. Give it a clear run name, for example:
    ```bash
    python train.py train-target --run-name "cnn_initial_test" --target-model "TargetCNN" --epochs 10
    ```
2.  **Execute Phase 2:** Run `train.py train-optimizer` to train the `PerceiverOptimizer` on the trajectory you just created. Use 'permutation' mode for 20 epochs.
    ```bash
    python train.py train-optimizer --run-name "cnn_initial_test" --optimizer-model "PerceiverOptimizer" --epochs 20 --training-mode "permutation"
    ```
3.  **Verify Checkpointing:** Confirm that checkpoint files are being saved in the `checkpoints/` directory.
4.  **Verify Resuming:** Run the Phase 2 command again, but this time for 10 more epochs, using the `--resume-from` flag to point to the last saved checkpoint.
    ```bash
    python train.py train-optimizer --run-name "cnn_initial_test" --epochs 10 --resume-from "checkpoints/PerceiverOptimizer_epoch_19.safetensors"
    ```

Report on the successful completion of this protocol.

## 6. Implementation Constraints and Directives 

In addition to the tasks outlined above, you must strictly adhere to the following constraints during all implementation phases.

### 6.1 No Placeholders or Dummy Implementations

All code you write must be complete and fully functional. The use of placeholders, stubs, or dummy code is strictly forbidden.

* Do not use comments like `# TODO: Implement later`.
* Do not implement functions or methods that simply `pass` or `raise NotImplementedError`.
* Every piece of code must be ready to execute and contribute to the successful completion of the testing protocol outlined in Section 5.

### 6.2 Adaptation Over Reinvention

Your primary strategy must be to adapt and integrate the existing, high-quality code within this repository. Do not write new implementations of components that are already present and functional. This is a waste of time and computational resources.

* **The Perceiver IO Model**: This directive applies most critically to the Perceiver IO model. You **must not** attempt to reimplement the Perceiver architecture from scratch. You are required to import the `PerceiverIO` class directly from the provided source file and wrap it within the new `PerceiverOptimizer` class as specified in Task 4.3.
* **The Target CNN**: Similarly, the `TargetCNN` is already defined and should be imported and used as-is.
* **The Exception**: The main exception to this rule is the new master script, `train.py`. This script is intended to be a new file that orchestrates and controls the various adapted components from a central location.

## 7. Principle II Reiteration: Maximization of Information Density

Every element of a generated artifact must serve a necessary and sufficient purpose. All redundant, tautological, or conversational elements must be eliminated. Comments are permissible only if they provide critical, non-obvious information required for operation.

## 8. Principle III Reiteration: Purpose-Driven Utility & Safety

All outputs are functional components of a research apparatus. Modern standards such as `safetensors` must be used for serialization, and the code should assume availability of necessary libraries like `torch`.

