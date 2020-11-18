import gzip
import json
import ast
import numpy as np
import pandas as pd

limit = 10**2
i = 0
#reviewer_list = []
#item_list = []
d = {}
file = gzip.open('reviews_Musical_Instruments.json.gz','r')
for line in file:
    if i > limit:
        break
    dict_str = line.decode("UTF-8")
    mydata = ast.literal_eval(dict_str)
    reviewerid = mydata["reviewerID"]
    itemid = mydata["asin"]
    if reviewerid not in d:
        d[reviewerid] = [itemid]
    if reviewerid in d:
        d[reviewerid].append(itemid)
    i += 1

print(d)
# this prints the dictionary in the following form
'''
{'A1YS9MDZP93857': ['0006428320', '0006428320'], 'A3TS466QBAWB9D': ['0014072149', '0014072149'], 'A3BUDYITWUSIS7': ['0041291905', '0041291905'],
'A19K10Z0D2NTZK': ['0041913574', '0041913574'], 'A14X336IB4JD89': ['0201891859', '0201891859'], 'A2HR0IL3TC4CKL': ['0577088726', '0577088726'], 
'A2DHYD72O52WS5': ['0634029231', '0634029231'], 'A1MUVHT8BONL5K': ['0634029347', '0634029347', '0634029355'], 'A15GZQZWKG6KZM': ['0634029347', '0634029347'],
'A16WE7UU0QD33D': ['0634029347', '0634029347', '0634029363'], 'AXMWZYP2IROMP': ['0634029355', '0634029355'], 'A6DCKXX4659CR': ['0634029355', '0634029355'],
'A28YJZCV43ZWQW': ['0634029355', '0634029355'], 'A2I4CV4PGZCNAF': ['0634029355', '0634029355'],...}
'''
# the following commented code will convert an output to excel file

#df = pd.DataFrame (_some_np_arr)
#filepath = 'output.xlsx'
#df.to_excel(filepath, index=False)

# the data is available here
#http://jmcauley.ucsd.edu/data/amazon/links.html
