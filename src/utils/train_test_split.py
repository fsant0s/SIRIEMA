from config.definitions import ROOT_DIR, RNDN
import sys
sys.path.append(ROOT_DIR)

import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(ROOT_DIR + "/datasets/raw/default_credit_card_clients/UCI_Credit_Card.csv")
dataset.drop(["ID"], axis = 1, inplace = True)
dataset.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
dataset.rename(columns={'PAY_0':'PAY_1'}, inplace=True)

train_data, val_data = train_test_split(dataset, test_size=0.2, random_state = RNDN)
val_data, test_data = train_test_split(val_data, test_size=0.5, random_state = RNDN)

pd.DataFrame(train_data).to_csv(ROOT_DIR + "/datasets/processed/UCI_Credit_Card_train.csv", index = False)
pd.DataFrame(val_data).to_csv(ROOT_DIR + "/datasets/processed/UCI_Credit_Card_val.csv", index = False)
pd.DataFrame(test_data).to_csv(ROOT_DIR + "/datasets/processed/UCI_Credit_Card_test.csv", index = False)

print("train", train_data.shape)
print("val", val_data.shape)
print("test", test_data.shape)
