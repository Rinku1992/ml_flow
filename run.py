import os

n_estimator = [100,150,200,250]
max_depth = [5,10,15,20]

for n in n_estimator:
    for m in max_depth:
        # print(n,m)
        os.system(f"python basic_ml_model.py -n{n} -m{m}")

