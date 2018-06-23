
# coding: utf-8

# In[1]:


import timeit
import tqdm
import pprint
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 'large'


# In[3]:


import pyquaternion
import quaternion
import rowan


# In[6]:


def arr_to_pyquat(arr):
    if len(arr.shape) > 1:
        pq_arr = np.empty(arr.shape[:-1], dtype='object')
        for i, x in enumerate(arr):
            pq_arr[i] = pyquaternion.Quaternion(x)
    else:
        pq_arr = np.array([pyquaternion.Quaternion(arr)])
    return pq_arr

def arr_to_npquat(arr):
    return quaternion.as_quat_array(arr)

pyquat_times = {}
quat_times = {}
rowan_times = {}
#max_log_N = 7
#Ns = [10**i for i in range(max_log_N)]
Ns = [10, 100000]
num = 10
pqlim = 1e8


# In[5]:


pyquat_times['Multiply'] = []
quat_times['Multiply'] = []
rowan_times['Multiply'] = []
for N in tqdm.tqdm(Ns):
    x = rowan.random.rand(N)
    y = rowan.random.rand(N)

    if N < pqlim:
        pyquat_times['Multiply'].append(
            timeit.timeit(
                "x*y",
                setup="from __main__ import x, y, arr_to_pyquat; x = arr_to_pyquat(x); y = arr_to_pyquat(y)",
                number = num
            )
        )
    quat_times['Multiply'].append(
        timeit.timeit(
            "x*y",
            setup="from __main__ import x, y, arr_to_npquat; x = arr_to_npquat(x); y = arr_to_npquat(y)",
            number = num
        )
    )
    rowan_times['Multiply'].append(
        timeit.timeit(
            "rowan.multiply(x, y)",
            setup="from __main__ import x, y, rowan",
            number = num
        )
    )


# In[6]:


pyquat_times['Exponential'] = []
quat_times['Exponential'] = []
rowan_times['Exponential'] = []
for N in tqdm.tqdm(Ns):
    x = rowan.random.rand(N)

    if N < pqlim:
        pyquat_times['Exponential'].append(
            timeit.timeit(
                "for i in range(len(x)): pyquaternion.Quaternion.exp(x[i])",
                setup="from __main__ import x, pyquaternion, arr_to_pyquat; x = arr_to_pyquat(x);",
                number = num
            )
        )
    quat_times['Exponential'].append(
        timeit.timeit(
            "np.exp(x)",
            setup="from __main__ import x, arr_to_npquat, np; x = arr_to_npquat(x);",
            number = num
        )
    )
    rowan_times['Exponential'].append(
        timeit.timeit(
            "rowan.exp(x)",
            setup="from __main__ import x, rowan",
            number = num
        )
    )


# In[7]:


pyquat_times['Conjugate'] = []
quat_times['Conjugate'] = []
rowan_times['Conjugate'] = []
for N in tqdm.tqdm(Ns):
    x = rowan.random.rand(N)

    if N < pqlim:
        pyquat_times['Conjugate'].append(
            timeit.timeit(
                "for i in range(len(x)): x.conjugate",
                setup="from __main__ import x, arr_to_pyquat; x = arr_to_pyquat(x);",
                number = num
            )
        )
    quat_times['Conjugate'].append(
        timeit.timeit(
            "x.conjugate()",
            setup="from __main__ import x, arr_to_npquat; x = arr_to_npquat(x);",
            number = num
        )
    )
    rowan_times['Conjugate'].append(
        timeit.timeit(
            "rowan.conjugate(x)",
            setup="from __main__ import x, rowan",
            number = num
        )
    )


# In[7]:


pyquat_times['Norm'] = []
quat_times['Norm'] = []
rowan_times['Norm'] = []
for N in tqdm.tqdm(Ns):
    x = rowan.random.rand(N)

    if N < pqlim:
        pyquat_times['Norm'].append(
            timeit.timeit(
                "for i in range(len(x)): x[i].norm",
                setup="from __main__ import x, arr_to_pyquat; x = arr_to_pyquat(x);",
                number = num
            )
        )
    quat_times['Norm'].append(
        timeit.timeit(
            "np.abs(x)",
            setup="from __main__ import x, np, arr_to_npquat; x = arr_to_npquat(x);",
            number = num
        )
    )
    rowan_times['Norm'].append(
        timeit.timeit(
            "rowan.norm(x)",
            setup="from __main__ import x, rowan",
            number = num
        )
    )


# In[9]:


pyquat_times['To Matrix'] = []
quat_times['To Matrix'] = []
rowan_times['To Matrix'] = []
for N in tqdm.tqdm(Ns):
    x = rowan.random.rand(N)

    if N < pqlim:
        pyquat_times['To Matrix'].append(
            timeit.timeit(
                "for i in range(len(x)): x[i].rotation_matrix",
                setup="from __main__ import x, arr_to_pyquat; x = arr_to_pyquat(x);",
                number = num
            )
        )
    quat_times['To Matrix'].append(
        timeit.timeit(
            "quaternion.as_rotation_matrix(x)",
            setup="from __main__ import x, quaternion, arr_to_npquat; x = arr_to_npquat(x);",
            number = num
        )
    )
    rowan_times['To Matrix'].append(
        timeit.timeit(
            "rowan.to_matrix(x)",
            setup="from __main__ import x, rowan",
            number = num
        )
    )


# In[10]:


pyquat_times['N'] = list(np.array(Ns)[np.array(Ns) < pqlim])
quat_times['N'] = Ns
rowan_times['N'] = Ns


# In[11]:


pp = pprint.PrettyPrinter(indent=4)
pp.pprint(pyquat_times)
pp.pprint(quat_times)
pp.pprint(rowan_times)


# In[12]:


df_pq = pd.DataFrame(pyquat_times).melt(id_vars="N", var_name="operation", value_name="pyquaternion")
df_nq = pd.DataFrame(quat_times).melt(id_vars="N", var_name="operation", value_name="npquaternion")
df_r = pd.DataFrame(rowan_times).melt(id_vars="N", var_name="operation", value_name="rowan")
df = df_r.merge(df_nq, on =["N", "operation"])
df = df.merge(df_pq, on =["N", "operation"], how = "left")
df.fillna(0, inplace=True)
df['pyquaternion'] /= df['N']
df['pyquaternion'] *= 1e6
df['npquaternion'] /= df['N']
df['npquaternion'] *= 1e6
df['rowan'] /= df['N']
df['rowan'] *= 1e6


# In[13]:


df[(df['N'] == 100000)].drop('pyquaternion', axis=1)
view = df.groupby(["N", "operation"]).mean()
view['rowan vs. npq'] = view['rowan']/view['npquaternion']
view['pyq vs. rowan'] = view['pyquaternion']/view['rowan']
print(view)


# In[14]:


cols = list(col['color'] for col in plt.rcParams['axes.prop_cycle'])
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
df[df['N'] == 10].drop(['N', 'pyquaternion'], axis=1).groupby(
    ["operation"]).mean().plot.barh(ax=axes[0, 0], logx=True, color = cols[0:2], title="$\log_{10}(N) = 1$")
df[df['N'] == 100000].drop(['N', 'pyquaternion'], axis=1).groupby(
    ["operation"]).mean().plot.barh(ax=axes[0, 1], logx=True, color = cols[0:2], title="$\log_{10}(N) = 6$")
df[df['N'] == 10].drop(['N', 'npquaternion'], axis=1).groupby(
    ["operation"]).mean().plot.barh(ax=axes[1, 0], logx=True, color = cols[0:4:2], title="$\log_{10}(N) = 1$")
df[df['N'] == 100000].drop(['N', 'npquaternion'], axis=1).groupby(
    ["operation"]).mean().plot.barh(ax=axes[1, 1], logx=True, color = cols[0:4:2], title="$\log_{10}(N) = 6$")
for ax in axes.flatten():
    ax.set_ylabel("")
plt.show()
fig.savefig("Performance.png")

