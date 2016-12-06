# -*- coding: utf-8 -*-

import os
import zipfile

import urllib.request
import tensorflow as tf

def download(url, filename):
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  return filename

def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
