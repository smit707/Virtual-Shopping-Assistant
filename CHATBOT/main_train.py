from Intents.Intents import new_intents
from ChatBot_v2.generate import DataGenerator
from ChatBot_v2.trainer import ModelTrainer

DataGenerator.generate(new_intents, dir_path='./data')

mt = ModelTrainer('./data/train_x_df.csv', './data/train_y_df.csv')
mt.startTraining('./model')