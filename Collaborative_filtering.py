# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:48:23 2019

@author: Administrator
"""

# ## Part 3: Collaborative filtering
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use(['seaborn-paper', 'fivethirtyeight', 'seaborn-whitegrid'])



# In[ ]:


def get_similarity(us_from_v, user_id):
    """output:
        a pd dataframe with columns: userId,simiarity score"""
    
    ans = []
    # data size
    u = len(us_from_v) # useId
    
    # this user watched movies
#     print(us_from_v)
    me_watched = pd.DataFrame(us_from_v[user_id])
#     print(me_watched)
    me_watched.columns = ['movieId', 'me_ratings']
    for i in range(u):
        # find overlapping movies, ratings
        i_watched = pd.DataFrame(us_from_v[i])
        i_watched.columns = ['movieId', 'i_ratings']
        overlap = pd.merge(me_watched, i_watched, how='inner', on=['movieId'])
             
        # computer similarity        
        o_center = overlap.apply(lambda x: x - overlap.mean(axis = 0), axis = 1)
        me_col = o_center['me_ratings']
        i_col = o_center['i_ratings']
        sim = np.sum(me_col*i_col)/np.sqrt(np.sum(me_col**2) * np.sum(i_col**2))
        
        # collect data
        ans_i = {'userId':i, 'sim': sim}
        ans.append(ans_i)
        
    # sort data by abs(similarity)
    ans_df = pd.DataFrame(ans)
    
    ans_df['abs_sim'] = ans_df['sim'].abs()
    ans_df.sort_values(['abs_sim'], ascending=False, inplace=True)
    ans_df.drop(['abs_sim'], axis = 1, inplace=True)
  

    return ans_df
    


# In[ ]:


def check_watched(movieId, userId, us_from_v):
    # convert data to dictionary: 
    # {1: {1: 2, 3: 1}, 2: {1: 1, 3: 5}}
    d = {}
    for i in range(len(us_from_v)):
        d_inner = {}
        for t in us_from_v[i]:
            d_inner[t[0]] = t[1]
        d[i] = d_inner 
        
    # check if in
    if movieId in d[userId]:
        rating = d[userId][movieId]
        return (True, rating)
    else:
        return (False, 0)
    
    


# In[ ]:


def get_watched(us_from_v, movieId, sim_group, sim_num):
    """return:
        dataframe with columns:['userId', 'sim', 'ave_ratings']"""
    
    # initialize ans
    watched = pd.DataFrame(columns=['userId', 'sim', 'ave_ratings'])
    count = 0
    
    # each user check if she watched this movie 
    for i in range(len(sim_group)):
        #print('------------')
        userId = sim_group.loc[i, 'userId']
        #print('i,userId:',i, userId)
        
        while check_watched(movieId, userId, us_from_v)[0] and count < sim_num+1:
            # get ['userId', 'sim', 'ave_ratings']
            watched.loc[count] = sim_group.loc[i]
            # get that rating
            watched.loc[count, 'movie_rating'] = check_watched(movieId, userId, us_from_v)[1]
            count = count + 1
            
            #print(watched)
            
    return watched
    


# In[ ]:


def convertData(data):
    # minimum id = 1, transform to 0
    min_id = min([t[0] for t in data])
    if min_id == 1:
        for i in range(len(data)):
            data[i] = (data[i][0]-1, data[i][1]-1, data[i][2])
    # convert data
    u = max([t[0] for t in data]) + 1 # useId
    us_from_v = [[] for i in range(u)]
    for (a, i, r) in data:
        us_from_v[a].append((i, r))
        
    return us_from_v


# In[ ]:


# for one user a, predict rating in movie i
def predict(data, user_id, movieId, sim_num=5):
    # convert Data
    us_from_v = convertData(data)
    
    # get similar people
    sim_group = get_similarity(us_from_v, user_id)
    
    # add average ratings for each user
    ave_ratings = [np.mean([t[1] for t in u]) for u in us_from_v]
    sim_group['ave_ratings'] = ave_ratings
        
    # get people who watched this movie
    watched_group = get_watched(us_from_v, movieId, sim_group, sim_num)
    
    # calculate scores
    weights = watched_group['sim']
    y_i = watched_group['movie_rating']
    ybar = watched_group['ave_ratings']
    deviation = np.sum((y_i - ybar) * weights) / weights.sum()
    
    # get this user's average rating
    rating_list = [t[1] for t in us_from_v[user_id]]
    user_average = np.mean(rating_list)
    
    pred_rating = user_average + deviation    
    return pred_rating


def main():
    ratings = pd.read_csv('ml-100k/u.data', delimiter = '\t', header=None)
    
    ratings.columns = ['userId', 'movieId', 'rates', 'timestamp']
    ratings = ratings.drop(['timestamp'], axis = 1)
    
    data = [tuple(i) for i in list(ratings.values)]
    print(predict(data, 1, 100, sim_num=5))


if __name__ == "__main__":
    import cProfile
    cProfile.run("main()")







