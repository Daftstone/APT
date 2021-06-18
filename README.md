# Fight Fire with Fire: Towards Robust Recommender Systems via Adversarial Poisoning Training

This project is for the paper "Fight Fire with Fire: Towards Robust Recommender Systems via Adversarial Poisoning Training".

The code was developed on Python 3.6 and tensorflow 1.15.0.

## Usage
### poionsing data preparation
```
mkdir poison_data
```
Put the poisoning data generated by attacks in this folder, and the poisoning data is named dataset_attack_type_attack_size.npy, e.g., ml-100k_random_30.npy.

### run main.py
```
usage: python main.py [--data DATA_NAME] [--gpu GPU_ID]
[--top_k TOP_K] [--extend ERM_USERS] [--target_index TARGET_ITEMS]

optional arguments:
  --dataset DATA_NAME
                        Supported: filmtrust, ml-100k, ml-1m, yelp.
  --gpu GPU_ID
                        GPU ID, default is 0.
  --top_k TOP_K
                        HR@top_k, default is 50.
  --extend ERM_USERS
                        The number of ERM users, default is 50.
  --target_index TARGET_ITEMS
                        The index of predefined target item list: 0, 1 for ml-100k, 2,3 for ml-1m, 4,5 for filmtrust, 6,7 for yelp.
```

### 4. Example.
```bash
python main.py --dataset ml-100k --gpu 0 --top_k 50 --extend 50 --target_index 0
```
