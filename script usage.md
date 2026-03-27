# Script Usage

```bash
.
|-- runModels.sh # Entry point to run the models and check the correctness
|-- server
|-- scripts/
    |-- test.sh
    |-- correctness_test.py
    |-- correctness_utils.py
    |-- genOpModels.py # Generate the op models
    |-- ops_meta.py    # Meta data of the ops
    |-- models.list    # List of the models to run, add extra model path here
```

## runModels.sh

Edit the configuration at the top of the script if needed:

```bash
# Config
PORT_A=8500 # You can change it to any free port
PORT_B=8501 # You can change it to any free port
export PORT_A PORT_B
BATCH_SIZE=10
ABS_TOL=1e-7 # Set atol
REL_TOL=1e-6 # Set rtol
```

Do correctness check for all models listed in `scripts/models.list`:

```bash
./runModels.sh
```

Do correctness check for one model (by short name):


``` bash
# e.g to run model  model_MATUL:
./runModels.sh MATMUL
```

Use `save` argument to only compile the model using xla
``` bash
./runModels.sh MATMUL save
```

Output files:

- `check_result.txt`: final summary.
- `tmp/`: logs and tf/xla dump files.
- `data.json` / `results.json`: runtime files used by the check flow.

## genOpModels.py

Generate all models defined in `ops_meta.py`:

__Note__: it will overwrite the models.list every time, there is a `DEFAULT_MODELS` map in the script, you can add extra model path there.

```bash
python genOpModels.py
```

Use `--op` to generate one specific model:

```bash
python genOpModels.py --op MATMUL
```
