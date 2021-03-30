import glob
import shutil
import os
import pandas as pd

df = pd.read_csv("pipeline/data/train.csv")

src_dir = "/home/laptop-obs-128/Downloads/Cassava Problem Solutions nad weights/CasavaProblem/cassava-leaf-disease-classification/train_images/"
dst_dir = "pipeline/data/train/"

for i,row in df.iterrows():
    path = src_dir+row["image_id"]
    shutil.copy(path,dst_dir)