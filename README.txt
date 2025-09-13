Enhanced Transformer Model with Algorithmic Task Evaluation

This project implements a flexible transformer model designed to test
different attention mechanisms and their ability to generalize on
algorithmic reasoning tasks. The primary focus is on understanding how
well models trained on shorter sequences can handle significantly longer
inputs without retraining.

The implementation extends the standard transformer architecture with
multiple positional encoding strategies (sinusoidal embeddings, rotary
embeddings, learned embeddings) and attention mechanisms (full
attention, ALiBi, Longformer-style sparse attention). It also supports
recurrence through memory augmentation, enabling experiments on
long-context modeling.

Two algorithmic tasks are used as evaluation benchmarks: - Sorting: The
model learns to take a random integer sequence and output its sorted
version. - Addition: The model learns to add digits organized in two 
sequences and produce the correct sum as output.

By training on relatively short sequences and testing on longer unseen
ones, the experiments highlight how different architectural choices
affect length generalization.

Features

-   Custom multi-head attention supporting ALiBi, RoPE and
    Longformer-style windowed attention.
-   Configurable transformer depth, hidden dimensions, number of heads
    and feedforward expansion.
-   Dataset generators for sorting and addition tasks with adjustable
    sequence lengths.
-   Full training loop with gradient tracking, learning rate scheduling
    and warmup support.
-   Validation and generalization evaluation across multiple sequence
    lengths.
-   Visualization of learned attention maps per head and layer.
-   Automated experiment logging, reporting, and saving of results,
    plots and trained models.

Experiments can be defined as configurations specifying task type,
sequence length, model architecture and training parameters. The
framework runs each experiment end-to-end, generating comparative
reports and visualizations.

Models include: - Baseline: Standard full-attention
transformer with learned positional embeddings. - RoPE: Same model but
using rotary positional embeddings. - ALiBi: Transformer with sinusoidal
embeddings and ALiBi biasing in attention. - Longformer: Transformer
using sliding window sparse attention.

Results are saved with plots of training/validation curves, attention
maps, length generalization accuracy, and a CSV summary table comparing
all experiments.

Run the script directly to launch all predefined experiments. Each run
creates a timestamped experiment folder under ./experiments/, containing
configuration files, plots, trained weights, and reports.

For each experiment, the framework produces: - Training/validation loss
and accuracy plots. - Gradient norm plots for stability analysis. -
Heatmaps of attention patterns across layers and heads. - Length
generalization evaluation at multiple test sequence lengths. -
Performance summary report in both JSON and text format. - Trained model
checkpoint (.pth file). - A comparative analysis across all experiments.

This implementation is primarily pedagogically-oriented. It provides a
sandbox for testing transformer variants on tasks where exact
algorithmic generalization is required. Beyond sorting and addition, the
framework can be extended to other symbolic or algorithmic tasks, making
it useful for studying inductive biases and scaling properties of
attention mechanisms.

Notes

The streamlit app is the most refined version of the code. The jupyter notebooks 
can be used for testing purpose.

