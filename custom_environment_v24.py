#this is for multi-target-tracking
#import the environment
from env import mtt_rwd_training as training
from env import mtt_rwd_eval as evaluation


trainingEnv = training.CustomEnvironment()
evalEnv = evaluation.CustomEnvironment()


