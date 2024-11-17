#this is for multi-target-tracking with obstacles
#import the environment
from env import mtt_rwd_WO_training as training
from env import mtt_rwd_WO_eval as evaluation


trainingEnv = training.CustomEnvironment()
evalEnv = evaluation.CustomEnvironment()


