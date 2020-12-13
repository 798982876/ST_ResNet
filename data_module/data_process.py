import data_module.quadtree as quadtree
import pandas as pd
import calendar
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
'''
本部分代码是对数据处理的一个过程
'''



def change_out_np(out_data,r1,c1):
    max_v = out_data.max(axis=0)
    min_v = out_data.min(axis=0)
    max_temp = int(max_v.tolist()[0])
    min_temp = int(min_v.tolist()[0])
    out_data_list = np.array(out_data).tolist()
    out_np = np.ones(shape=(int(r1), int(c1)))
    out_data_out = []
    for item in out_data_list:
        temp_out = (item[0] - min_temp) / (max_temp - min_temp)
        temp_np = out_np * temp_out
        temp_list = temp_np.tolist()
        for i in range(24):
            out_data_out.append(temp_list)
    return out_data_out
# 转矩阵并保存 
def turn_matrix(data_from_hour_fill, rows, cols):
    length = len(data_from_hour_fill)
    matrix_return_data = np.zeros((length, rows, cols))
    # matrix_return_quadtree = np.zeros((length, rows, cols),dtype = np.object)#保存四叉树

    data_key = sorted(data_from_hour_fill)
    i = 0
    for item in data_key:
        for data_value in data_from_hour_fill[item]:
            temp_row_value = data_value[3]
            temp_col_value = data_value[4]
            matrix_return_data[i][int(temp_row_value)][int(temp_col_value)] = int(
                data_value[2])
            # matrix_return_quadtree[i][int(temp_row_value)][int(temp_col_value)] = data_value[0]
            
        i = i + 1
    return matrix_return_data


# 获取缺失的时间，直接对按照小时统计的时间进行填充
def fill_timecode(time_start, time_end, timecode, timecode_all):
    start_index = timecode_all.index(str(time_start))
    end_index = timecode_all.index(str(time_end)) + 1
    timecode_all = timecode_all[start_index:end_index]
    timecode_no = []
    for item in timecode_all:
        if int(item) not in timecode:
            timecode_no.append(int(item))
    return timecode_no


# 通过timecode_no直接进行填充
def fill_date(data_from_hour, timecode_no):
    data_no = [0] * 5
    data_no_all = []
    data_no_all.append(data_no)
    for item in timecode_no:
        data_from_hour[str(item)] = data_no_all
    return data_from_hour


# 按照时间来进行统计数据 得到数据一共5列，四叉树编码、时间、数量、行号、列号
def get_data_by_hour(data, t):
    rehour = defaultdict(list)
    t = sorted(t)
    for h in tqdm(t):
        for item in data:
            if item[1] == h:
                rehour[h].append(item)
    return rehour


# 获取时间顺序表
def get_hours(data):
    t = set()
    for row in data:
        t.add(row[1])
    return t


# 改变数据从行列号从0开始
def change_data_from_zero(data, minr, minc):
    data_return = []
    for item in data:
        item[3] = item[3] - minr
        item[4] = item[4] - minc
        data_return.append(item)
    return data_return


# 获取行列号和最大最小行列号
def get_rows_cols(data_from_groupby):
    r = []
    c = []
    # 获取行号列号
    for row in data_from_groupby:
        r.append(row[3])
        c.append(row[4])
    maxr = max(r)
    minr = min(r)
    maxc = max(c)
    minc = min(c)

    r1 = maxr - minr + 1
    c1 = maxc - minc + 1
    return (r1, c1, minr, minc)


# 对数据库获取的数据进行聚类，最终得到的结果是四叉树编码，时间，数量，行号，列号
def groupby_data(data_from_pg, layer):
    data_from_pg_layer = process_data(data_from_pg, layer)
    data_from_pg_pd = pd.DataFrame(
        data_from_pg_layer,
        columns=['id', 'hour', 'longitude', 'latitude', 'quadtree_code'])
    x = data_from_pg_pd['id'].groupby(
        [data_from_pg_pd['quadtree_code'], data_from_pg_pd['hour']]).count()
    x_reset = x.reset_index()
    x_np = np.array(x_reset)
    x_np_list = x_np.tolist()
    #用来存储带有行列号的数据，一个6列数据，分别是hour,qudatree_code,count,row,column
    data_from_layer_cr = []
    for item in x_np_list:
        column_num, row_num = get_cols_and_rows(item[0])
        item.append(row_num)
        item.append(column_num)
        data_from_layer_cr.append(item)
    return data_from_layer_cr


# 对取到的时间 位置数据进行某一层级的四叉树编码，并返回数组数据
def process_data(data_from_pg, layer):
    data_from_process = []

    for item in data_from_pg:
        item = list(item)
        item.append(quadtree.geodetic2quadtree(item[2], item[3], layer))
        data_from_process.append(item)
    return data_from_process


# 根据四叉树编码获取求其行号和列号
def get_cols_and_rows(quadtree_code):
    quadtree_code_temp = quadtree_code[1:]
    quadtree_code_temp_length = len(quadtree_code_temp)
    c_temp = pow(2, quadtree_code_temp_length)
    r_temp = pow(2, quadtree_code_temp_length)
    c_temp_begin = 0
    r_temp_begin = 0
    c_temp_end = pow(2, quadtree_code_temp_length)
    r_temp_end = pow(2, quadtree_code_temp_length)
    i = 1
    while (i <= quadtree_code_temp_length):
        q_temp = quadtree_code_temp[i - 1:i]
        if q_temp == 'q':
            c_temp_end = (c_temp_begin + c_temp_end) / 2
            r_temp_end = (r_temp_begin + r_temp_end) / 2
            i = i + 1
        elif q_temp == 'r':
            c_temp_end = (c_temp_begin + c_temp_end) / 2
            r_temp_begin = (r_temp_begin + r_temp_end) / 2
            i = i + 1
        elif q_temp == 't':
            c_temp_begin = (c_temp_begin + c_temp_end) / 2
            r_temp_end = (r_temp_begin + r_temp_end) / 2
            i = i + 1
        elif q_temp == 's':
            c_temp_begin = (c_temp_begin + c_temp_end) / 2
            r_temp_begin = (r_temp_begin + r_temp_end) / 2
            i = i + 1
    cs = c_temp_end
    rs = r_temp_end
    return cs, rs


# 根据选取的时间范围，获取所有的时间顺序表，用来后续填充时间空白
def get_timecode_all(time_year):
    year = str(time_year)
    time_code = []
    for month in range(1, 13):
        if month < 10:
            month = '0' + str(month)
        month = str(month)
        monthRange = calendar.monthrange(int(year), int(month))
        dayCount = monthRange[1] + 1
        for day in range(1, dayCount):
            if day < 10:
                day = '0' + str(day)
            day = str(day)
            for hour in range(24):
                if hour < 10:
                    hour = '0' + str(hour)
                hour = str(hour)
                time_code.append(year + month + day + hour)
    return time_code
