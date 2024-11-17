#this is for multi-target-tracking
#import the environment
from env import multi_target_tracking_training_moving_targets as training
from env import multi_target_tracking_eval_moving_targets as evaluation


trainingEnv = training.CustomEnvironment()
evalEnv = evaluation.CustomEnvironment()


