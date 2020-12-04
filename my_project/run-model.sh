DATA_DIR=dataset
DATASET_NAME=final_data_set_3.csv
#DATASET_NAME=final_data_set_2.csv
#DATASET_NAME=final_data_set_1.csv

OUTPUT_FILE=output.csv
MEDICATION_FILE=medication_names.csv
python test-torch.py \
  --batch_size 128 \
  --no_of_epochs 20 \
  --dataset $DATA_DIR/$DATASET_NAME \
  --output $DATA_DIR/$OUTPUT_FILE \
  --medication $DATA_DIR/$MEDICATION_FILE \
