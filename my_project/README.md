# cmsc-516: Project

Used technology for the model: PyTorch, SpaCy, scikit-learn, pandas.

How to Setup:

## Software Requirements:

```
blis==0.7.3
catalogue==1.0.0
certifi==2020.11.8
chardet==3.0.4
cymem==2.0.4
dataclasses==0.6
en-core-web-sm==2.3.1
future==0.18.2
idna==2.10
joblib==0.17.0
murmurhash==1.0.4
numpy==1.19.4
pandas==1.1.4
plac==1.1.3
preshed==3.0.4
python-dateutil==2.8.1
pytz==2020.4
requests==2.25.0
scikit-learn==0.23.2
scipy==1.5.4
six==1.15.0
spacy==2.3.4
srsly==1.0.4
thinc==7.4.3
threadpoolctl==2.1.0
torch==1.7.0
torchtext==0.8.0
tqdm==4.54.0
typing-extensions==3.7.4.3
urllib3==1.26.2
wasabi==0.8.0
```

Then, please run this instruction on your environment/virtual environment:

`python -m spacy download en`

This will download the SpaCy dictionary which is necessary to run the project.

## Project Structure

The project includes the following files and folders:

  - __/dataset__: A folder that contains inputs that are used for the experiments. The model also saves here.
	- final_data_set_1.csv: CSV file that contains 52267 tweets with 200 positive classes
	- final_data_set_2.csv: CSV file that contains 52267 tweets with 500 positive classes
	- final_data_set_3.csv: CSV file that contains 52267 tweets with 1000 positive classes
	- medication_names.csv: CSV file that contains a set of precompiled medication names
    - output.csv: This CSV file will contain the output of the experiment.
  - test-tensor.py: this is the script for the model
  - run_model.sh: The entry point of the experiment



## Running Experiments
Step 1: Install software requirements mentioned above.

Step 2: Update the filepaths and parameters in *run_model.sh*

Step 3: `.run_model.sh`