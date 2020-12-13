import psycopg2
import pandas as pd
import os
# import json
'''
本部分代码是全程获取数据的部分
读取本地数据或者对数据库进行操作
'''

def get_space_extend(train_id):
    get_space_str = 'select train_condition#>\'{space_extend}\' from task.train where train_id = \'' + str(
        train_id) + '\';'
    space_extend = operate_task(get_space_str)
    east = space_extend[0][0]['east']
    south = space_extend[0][0]['south']
    west = space_extend[0][0]['west']
    north = space_extend[0][0]['north']
    return east, south, west, north


def get_time_extend(train_id):
    get_time_str = 'select train_condition#>\'{time_extend}\' from task.train where train_id = \'' + str(
        train_id) + '\';'
    time_extend = operate_task(get_time_str)
    start_time = time_extend[0][0]['start_time']
    end_time = time_extend[0][0]['end_time']
    return start_time, end_time


def insert_layer_info(layer, train_id, r1, c1, minr, minc):
    if layer == 10:
        insert_str = 'update task.train set train_condition = jsonb_insert(train_condition,\'{layer_info}\',\'{\"layer' + str(
            layer) + '\":{\"r\":' + str(r1) + ',\"c\":' + str(
                c1) + ',\"minr\":' + str(minr) + ',\"minc\":' + str(
                    minc) + '}}\') where train_id =\'' + str(
                        train_id) + '\' returning *'
    else:
        insert_str = 'update task.train set train_condition = jsonb_insert(train_condition,\'{layer_info,layer' + str(
            layer) + '}\',\'{\"r\":' + str(r1) + ',\"c\":' + str(
                c1) + ',\"minr\":' + str(minr) + ',\"minc\":' + str(
                    minc) + '}\') where train_id =\'' + str(
                        train_id) + '\' returning train_condition'
    print(insert_str)
    train_condition = operate_task(insert_str)
    return train_condition





def operate_task(query_str):
    conn = psycopg2.connect( 
        database="geotask",
        user="postgres",
        password="postgres",
        host="132.1.11.158",
        port="5432")
    cur = conn.cursor()
    cur.execute(query_str)
    conn.commit()
    rows = cur.fetchall()
    conn.close()
    return rows


def operate_jqb(query_str):
    conn = psycopg2.connect(
        database="crimeanalysis",
        user="crimeanalysis",
        password="crimeanalysis",
        host="132.1.11.158",
        port="5432")
    cur = conn.cursor()
    cur.execute(query_str)
    rows = cur.fetchall()
    return rows


def get_condition(train_id):
    query_condition_str = 'select train_condition from task.train where train_id =\'' + str(
        train_id) + '\''
    condition_by_train_id = list(operate_task(query_condition_str)[0])[0]
    start_time = condition_by_train_id['start_time']
    end_time = condition_by_train_id['end_time']
    west = condition_by_train_id['west']
    east = condition_by_train_id['east']
    south = condition_by_train_id['south']
    north = condition_by_train_id['north']
    return start_time, end_time, east, south, west, north


# 根据条件直接从数据库中获取数据并保存至本地
def get_data_from_pg(train_id):
    start_time, end_time = get_time_extend(train_id)
    east, south, west, north = get_space_extend(train_id)
    # 定义警情表数据查询语句
    query_string_by_time_space = "select a101,substring(b105,1,10) as time,st_X(geometry(B205)),st_Y(geometry(B205)) from crime.警情表 where substring(b105,1,10)>=\'" + str(
        start_time) + "\' and substring(b105,1,10)<=\'" + str(
            end_time) + "\' and " + str(
                west) + "<=st_X(geometry(B205)) and " + str(
                    east) + ">=st_X(geometry(B205)) and " + str(
                        north) + " >= st_Y(geometry(B205)) and " + str(
                            south) + " <=st_Y(geometry(B205))"
    print(query_string_by_time_space)
    data_from_jqb = operate_jqb(query_string_by_time_space)
    print(len(data_from_jqb))
    data_from_pd = pd.DataFrame(data_from_jqb)
    path = './data/ST_ResNet/train_' + str(train_id)
    print(path)
    # 路径不存在时就创建
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/' + str(train_id) + '.csv'
    # 会覆盖数据
    data_from_pd.to_csv(path, header=0, index=0)
    return 1


# 读取数据
def get_data_from_file(file):
    if os.path.exists(file):
        data_from_file = pd.read_csv(file,header=None)
        return data_from_file
    else:
        return 0

#ZJ:获取各层数据信息至本地json
# def get_layer_info(train_id):
    # get_info_str = 'select train_condition#>\'{layer_info}\' from task.train where train_id = \'' + str(
        # train_id) + '\';'
    # layer_info = operate_task(get_info_str)
    # print(layer_info)

    # root_path = "data/ST_ResNet/train_" + str(train_id) + "/"
    # temp_path = root_path + 'temp'
    # temp_name = temp_path + "/" + str(train_id)
    # with open(temp_name+ '_layerinfo.json', 'w') as outfile:
        # json.dump(layer_info[0][0], outfile, ensure_ascii=False)
        # outfile.write('\n')
    # return 0