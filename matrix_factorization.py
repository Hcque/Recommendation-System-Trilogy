# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:47:13 2019

@author: Administrator
"""

# ## Part 1: Matrix factorization
# 
# First we'll look in detail at one popular implementation of matrix factorization. It is the alternating least squares algorithm.
# We will assume that the input data is the forms of (a, i, r) triples, where a is a user index. i is an item index and r is a rating. Here is a small example:
# [(0, 0, 5), (0, 1, 3), (0, 3, 1),
#  (1, 0, 4), (1, 3, 1), 
#  (2, 0, 1), (2, 1, 1), (2, 3, 5), 
#  (3, 0, 1), (3, 3, 4), 
#  (4, 1, 1), (4, 2, 5), (4, 3, 4)].
#  
# 
# 
# 
import numpy as np

import matplotlib.pyplot as plt
plt.style.use(['seaborn-paper', 'fivethirtyeight', 'seaborn-whitegrid'])

# ### 1) Roadmap
#  The overall plan for als function is following:
# * Define n and m from the data.
# * Initialize a list of lists that indicates for each item the indices of users that rate that item and vice versa.
# * Initialize the set of parameters, note that the u,v entries are set randomly while the user and item offsets are set to 0.
# * Then we alternate minimizations.(update_U, update_V)
# * And, report the results, the error between predicted scores and a held-out set of actual scores on the same users and items.
# 
# Let us first define ridge regression.

# In[15]:


def ridge_analytic(X,Y,lam):
    n, d = X.shape
    th = np.linalg.solve(X.T @ X + lam* np.identity(d), X.T @ Y)
    return th


# In[16]:


def update_U(data, us_from_v, x, k, lam):
    u, v = x
    idx_list = [[t[0] for t in user] for user in us_from_v]
    rating_list = [[t[1] for t in user] for user in us_from_v]
    
    for i in range(len(u)):
        idx_movie = idx_list[i]
        rating = rating_list[i]    
        X = np.vstack([np.reshape(v[j], (1,k)) for j in idx_movie])
        y = rating
        
        u[i] = ridge_analytic(X,y,lam)
    x = u, v

    return x


# In[17]:


def update_V(data, vs_from_u, x, k, lam):
    u, v = x
    idx_list = [[t[0] for t in m] for m in vs_from_u]
    rating_list = [[t[1] for t in m] for m in vs_from_u]

    for i in range(len(v)):
        idx_user = idx_list[i]
        rating = rating_list[i]               
   
        X = np.vstack([np.reshape(u[j], (1,k)) for j in idx_user])
        y = rating
        v[i] = ridge_analytic(X,y,lam)
    x = u, v
#     print(x)
    return x


# In[18]:


def retrive_rating(x, a, i):
    U = x[0]
    V = x[1]
    rating = np.inner(U[a], V[i])
    return rating


# In[19]:


def rmse(x, us_from_v, lam):
    ans = 0
    
    # square loss
    all_error = 0
    for u_id in range(len(us_from_v)):
        for t in us_from_v[u_id]:
            m_id = t[0]
            m_rating = t[1]
            
            pred_rating = retrive_rating(x, u_id, m_id)
            error = (m_rating-pred_rating)**2
            all_error = all_error + error
    
    # regulazation terms
    u = len(x[0])
    m = len(x[1])
    k = len(x[0][0])
    U = np.reshape(np.array(x[0]), (u,k))
    V = np.reshape(np.array(x[1]), (m,k))
    penality_U = np.sum(U**2) * lam
    penality_V = np.sum(V**2) * lam
    
    # add all
    ans = all_error + penality_U + penality_V
    
    return np.sqrt(ans)


# In[20]:


def als(data, k=2, lam=0.02, max_iter=10):
    """input: data is list of (userId, movieId, ratings),
        k is the rank of approximation matrix,
        lam is regulazation parameter,
        
        output: the estimated matrix: U@V.
        """
    # minimum id = 1, transform to 0
    min_id = min([t[0] for t in data])
    if min_id == 1:
        for i in range(len(data)):
            data[i] = (data[i][0]-1, data[i][1]-1, data[i][2])
    # data size
    u = max([t[0] for t in data]) + 1 # useId
    m = max([t[1] for t in data]) + 1 # movieId
    
    # entries put in [[user_1], [user_2],...], where user_1 = (rating moive one, rating movie two, ...)
    us_from_v = [[] for i in range(u)]
    vs_from_u = [[] for i in range(m)]
    for (a, i, r) in data:
        us_from_v[a].append((i, r))
        vs_from_u[i].append((a, r))
    # initialize guess of vs
    x = ([np.random.normal(1, size=(k,1)) for a in range(u)],          [np.random.normal(1, size=(k,1)) for i in range(m)])
    
    rmse_list = []
    for i in range(max_iter):
#         print('iter time: %i' %i)
        # update u_i
#         print('update U -----------------------------------------')
        x = update_U(data, us_from_v, x, k, lam)
        # update v_j
#         print('update V -----------------------------------------')
        x = update_V(data, vs_from_u, x, k, lam)
        
        # report rmse
        rmse_ = rmse(x, us_from_v, lam)
        rmse_list.append(rmse_)
        print('root mean square error: %f' %rmse_)
    print('final rmse: %f' %rmse_)
    return x, rmse_list
