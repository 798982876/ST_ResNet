import math

PI = 3.1415926535897931  # 圆周率
DtoR = PI / 180.0  #度转弧度系数
Rtod = 180.0 / PI  #弧度转度


# 地理坐标转墨卡托坐标
def geodetic2mercator(longitude, latitude):
    L = 0
    B = 0
    L = longitude * DtoR
    B = latitude * DtoR
    temp = (1.0 + math.sin(B)) / (math.cos(B))
    B = math.log(temp)
    return L, B


# 墨卡托坐标转四叉树
def mercator2quadtree(x, y, layer):
    temp_x = x
    temp_y = y
    target = 't'
    i = 1
    while (i <= layer):
        if temp_y > 0:
            if temp_x <= 0:
                target = target + 'q'
                temp_x = temp_x + PI / 2
                temp_y = temp_y - PI / 2
            else:
                target = target + 'r'
                temp_x = temp_x - PI / 2
                temp_y = temp_y - PI / 2
        elif temp_x > 0:
            target = target + 's'
            temp_x = temp_x - PI / 2
            temp_y = temp_y + PI / 2
        else:
            target = target + 't'
            temp_x = temp_x + PI / 2
            temp_y = temp_y + PI / 2
        temp_x = temp_x * 2
        temp_y = temp_y * 2
        i = i + 1
    return target


# 四叉树编码转墨卡托坐标
def quadtree2mercator(quadtree_code):
    left = -PI
    top = PI * 3
    right = PI * 3
    bottom = -PI
    quad = quadtree_code.lower()
    i = 1
    length = len(quadtree_code)
    while i <= length:
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        temp_quad = quad[i - 1:i]
        if temp_quad == 'q':
            right = center_x
            bottom = center_y
        elif temp_quad == 'r':
            left = center_x
            bottom = center_y
        elif temp_quad == 's':
            left = center_x
            top = center_y
        elif temp_quad == 't':
            right = center_x
            top = center_y
        i = i + 1
    return left, top, right, bottom


#墨卡托坐标转地理坐标
def mercatror2geodetic(x, y):
    L = x
    B = 2 * math.atan(math.exp(y)) - 0.5 * PI
    L = L * Rtod
    B = B * Rtod
    return L, B


# 地理坐标系转四叉树编码
def geodetic2quadtree(longitude, latitude, layer):
    x, y = geodetic2mercator(longitude, latitude)
    target = mercator2quadtree(x, y, layer)
    return target


# 四叉树编码转地理坐标
def quadtree2geodetic(quadtree_code):
    right, bottom, left, top = quadtree2mercator(quadtree_code)
    west, north = mercatror2geodetic(left, top)
    east, south = mercatror2geodetic(right, bottom)
    return east, south, west, north

