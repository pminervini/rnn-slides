#!/bin/bash

jupyter nbconvert --to slides --ServePostProessor.port=8910 --ServePostProcessor.open_in_browser=False --post serve Recurrent_Neural_Networks.ipynb
