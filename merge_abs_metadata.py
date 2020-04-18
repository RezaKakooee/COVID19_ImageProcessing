# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:23:56 2020

@author: reza
"""

import pandas as pd

abs_metadata_pub_df = pd.read_csv('abs_metadata_pub.csv')
abs_metadata_uog_df = pd.read_csv('abs_metadata_uog.csv')

abs_metadata_df = pd.concat([abs_metadata_pub_df, abs_metadata_uog_df])
abs_metadata_df.to_csv('abs_metadata.csv', index=False)