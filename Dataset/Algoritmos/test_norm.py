#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:02:06 2017

@author: kratos
"""

import util as util
import pandas as pd
import numpy as np

columns = ['f1', 'f2', 'Evadido', 'CodigoTurma']

test = [
        (1,5,0,10),
        (5,5,0,10),
        (7,5,1,10),
        (2,5,0,15),
        (7,5,1,15),
        (9,5,1,15),
        (3,5,0,15)
        ]

df = pd.DataFrame.from_records(test, columns=columns)

df_norm = util.normalize(df, ['Evadido', 'CodigoTurma'])