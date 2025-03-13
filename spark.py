import sys
import os
from os.path import dirname, realpath
sys.path.append(os.path.join(dirname(realpath(__file__)), "sparktts"))
from models.SparkTTS import SparkTTS

model = SparkTTS(os.environ["HF_HOME"], "cuda")

print(model)