#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:27:13 2017

@author: kratos
"""

def calcular_z_score(dataframe):
    for col in dataframe:
        dataframe[col] = (dataframe[col] - dataframe[col].mean())/dataframe[col].std(ddof=0)
    return dataframe;