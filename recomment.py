import os
import sys
import json
import numpy as np


def count(userdate, user1, user2):
    houses = set()
    for house in userdate[user1]:
        if house in userdate[user2]:
            houses.add(house)
    n = len(houses)
    if n == 0:
        return 0
    x = np.array([userdate[user1][move] for move in houses])
    y = np.array([userdate[user2][move] for move in houses])
    sx = x.sum()
    sy = y.sum()
    xx = (x**2).sum()
    yy = (y**2).sum()
    xy = (x * y).sum()
    sxx = xx - sx**2 / n
    syy = yy - sy**2 / n
    sxy = xy - sx * sy / n
    if sxx * syy == 0:
        return 0
    pearson_score = sxy / np.sqrt(sxx * syy)
    return pearson_score


def read_data(filename):
    with open(filename, 'r') as f:
        ratings = json.loads(f.read())
    return ratings


def count_ps(userdate):
    '''计算各个用户间的皮氏分数'''
    users = list(userdate.keys())
    psmat = []
    for user1 in users:
        psrow = []
        for user2 in users:
            psrow.append(count(userdate, user1, user2))
        psmat.append(psrow)
    users = np.array(users)
    psmat = np.array(psmat)
    return users, psmat


def find_similars(users, psmat, user, n_similars=None):
    '''查找相似用户
    similar_users;相似用户
    similar_scores；相似度'''
    user_index = np.arange(len(users))[users == user][0]
    sorted_indices = psmat[user_index].argsort()[::-1]
    similar_indices = sorted_indices[
        sorted_indices != user_index][:n_similars]
    similar_users = users[similar_indices]
    similar_scores = psmat[user_index][similar_indices]
    return similar_users, similar_scores


def recomment(userdate, user):
    users, psmat = count_ps(userdate)
    similar_users, similar_scores = find_similars(
        users, psmat, user)
    a = similar_scores > 0
    similar_users = similar_users[a]
    similar_scores = similar_scores[a]
    score_sums, weight_sums = {}, {}#根据用户相似度和房源质量进行打分排序
    for i, similar_user in enumerate(similar_users):
        for house, score in userdate[similar_user].items():
            if house not in userdate[user].keys() or \
                    userdate[user][house] == 0:
                if house not in score_sums.keys():
                    score_sums[house] = 0
                score_sums[house] += score * similar_scores[i]
                if house not in weight_sums.keys():
                    weight_sums[house] = 0
                weight_sums[house] += similar_scores[i]
    house_ranks = {house: score_sum / weight_sums[house]
                   for house, score_sum in score_sums.items()}
    sorted_indices = np.array(
        list(house_ranks.values())).argsort()[::-1]
    recolist = np.array(
        list(house_ranks.keys()))[sorted_indices]
    return recolist


def write(item):
    with open("recolist.json", "ab") as f:
        text = json.dumps(dict(item), ensure_ascii=False) + '\n'
        f.write(text.encode('utf-8'))
        print("writeOK")

def main(argc, argv, envp):
    item = {}
    userdate = read_data('usersdate.json')
    for user in userdate.keys():
        recolist = recomment(userdate, user)
        print('{}: {}'.format(user, recolist))
        item[user]=recolist.tolist()
    write(item)
    return None


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
