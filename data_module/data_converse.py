import data_module.data_get as data_get
import data_module.data_process as data_process
import pandas as pd
import numpy as np
import os
import json

'''
本部分代码 是一个流程，将数据从本地读取到转换为填充时间之后的npy数据
'''


def data_to_np(train_id, layer):
    root_path = "data/ST_ResNet/train_" + str(train_id) + "/"
    csv_file = root_path + str(train_id) + '.csv'
    out_csv_file = 'data/ST_ResNet/out.csv'
    # 判断是否存在该文件
    if not os.path.exists(csv_file):
        return 0
    elif not os.path.exists(csv_file):
        return 0
    else:
        # 根据csv数据地址读取csv 并转为list数据，方便后续处理
        data_out_from_file = data_get.get_data_from_file(out_csv_file)
        data_from_file = data_get.get_data_from_file(csv_file)
        data_from_file = np.array(data_from_file)
        data_from_file = data_from_file.tolist()
        # 根据读取的数据和层级进行数据聚类划分
        data_from_groupby = data_process.groupby_data(data_from_file, layer)
        # 根据聚类的数据 获取其行列号，以及最小的行列号
        (r1, c1, minr, minc) = data_process.get_rows_cols(data_from_groupby)
        # 存入数据库中的条件属性中 包含layer信息
        train_condition = data_get.insert_layer_info(layer, train_id, r1, c1,
                                                     minr, minc)
        # 将天气数据按照行列号，归一化之后，存入本地
        out_data_out = data_process.change_out_np(data_out_from_file, r1, c1)
        out_data_out = np.array(out_data_out)
        # 改变数据从行列号从0开始
        data_from_zero = data_process.change_data_from_zero(
            data_from_groupby, minr, minc)
        # 获取其所有的时间序列
        timecode_all = []
        # 获取起始时间字符串
        start_time, end_time = data_get.get_time_extend(train_id)
        start_year = start_time[0:4]
        end_year = end_time[0:4]
        for year in range(int(start_year), int(end_year) + 1):
            timecode = data_process.get_timecode_all(year)
            timecode_all = timecode_all + timecode
        timecode = data_process.get_hours(data_from_zero)
        # 获取数据为空白的时间序列
        timecode_no = data_process.fill_timecode(
            start_time[0:10], end_time[0:10], timecode, timecode_all)
        data_from_hour = data_process.get_data_by_hour(data_from_zero,
                                                       timecode)

        temp_path = root_path + 'temp'
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)

        temp_name = temp_path + "/" + str(train_id)
        with open(temp_name + '_' + str(layer) + '.json', 'w') as outfile:
            json.dump(data_from_hour, outfile, ensure_ascii=False)
            outfile.write('\n')

        with open(temp_name + '_' + str(layer) + '.json', 'r') as json_file:
            data_from_json = json.load(fp=json_file)
        # 填充数据
        data_from_hour_fill = data_process.fill_date(data_from_json,
                                                     timecode_no)
        # 将数据转为矩阵
        data_matrix_data= data_process.turn_matrix(data_from_hour_fill, int(r1),
                                               int(c1))
        # data_matrix_data,data_matrix_quadtree = data_process.turn_matrix(data_from_hour_fill, int(r1),int(c1))
        # print('data_matrix',data_matrix)
        # 存储为numpy文件
        numpy_path = root_path + 'input/' + str(layer)

        if not os.path.exists(numpy_path):
            os.makedirs(numpy_path)
        with open(numpy_path + '/' + str(train_id) + '_' + str(layer) + '.npy',
                  'wb') as numpy_file:
            np.save(numpy_file, data_matrix_data)
        with open(numpy_path + '/' + str(train_id) + '_' + str(layer) + '_out.npy',
                  'wb') as numpy_file:
            np.save(numpy_file, out_data_out)
        # with open(numpy_path + '/' + str(train_id) + '_' + str(layer) + '_quadtree.npy',
                  # 'wb') as numpy_file:
            # np.save(numpy_file, data_matrix_quadtree,allow_pickle=True)
        
        return 1
