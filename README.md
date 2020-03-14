# Pyanomaly: Autoencoder based anomaly detector

## Requirements

* Python 3.x
* Keras
* Pandas

## Trianing Mode

Train an autoencoder model to learn the identity function of some dataset.

### Syntax

```
python -m pyanomaly [-h] --mode training [--input INPUT] [--output OUTPUT]
                    [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE]
                    [--loss {mse}] [--layers LAYERS]
                    [--output-activation {sigmoid,relu}]
                    [--hidden-activation {sigmoid,relu}]

```
