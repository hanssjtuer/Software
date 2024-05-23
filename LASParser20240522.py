import os
# 判断路径是不是文件，文件夹路径是不是存在，创建文件夹路径，创建文件路径，获取文件夹中的全部文件，分解路径
import laspy
# 读取LAS文件，生成LAS文件
import csv
# 生成CSV文件
from shutil import copyfile
# 复制文件
import numpy as np
import glob
# 获取文件夹中的全部文件

if __name__ == "__main__":
    las_file_path = r"D:\杆塔GIM与点云模型构建代码加工20240501\220kV济鹤24B8线济桐24B7线1#-71#.las"
    if not os.path.isfile(las_file_path):
        print("不是文件")
        exit()
        
    if not las_file_path.endswith(".las"):
        print("不是LAS文件")
        exit()
        
    result_first_level_folder = las_file_path[::-1].replace(".las"[::-1], "", 1)[::-1] + "解析结果"
    # 删除LAS文件路径末尾的".las"
    if not os.path.exists(result_first_level_folder):
        os.mkdir(result_first_level_folder)
    print("LAS文件解析结果存储到如下文件夹：" + result_first_level_folder)
    
    result_second_level_las_folder = os.path.join(result_first_level_folder, "LAS文件分类（0-31）")
    if not os.path.exists(result_second_level_las_folder):
        os.mkdir(result_second_level_las_folder)
        
    if not os.path.exists(os.path.join(result_second_level_las_folder, "unassigned31.las")):
        las_file_context = laspy.read(las_file_path)
        for i in range(0, 32, 1):
            unassigneds = laspy.create(point_format=las_file_context.header.point_format, file_version=las_file_context.header.version)
            unassigneds.points = las_file_context.points[las_file_context.classification == i]
            unassigneds.header = las_file_context.header
            unassigneds.write(os.path.join(result_second_level_las_folder, "unassigned" + str(i)+ ".las"))
            
    las_info_csv_file_path = os.path.join(result_first_level_folder, "公用文件头块-变量长度记录-点数据记录.csv")
    if not os.path.exists(las_info_csv_file_path):
        las_file_context = laspy.read(las_file_path)
        las_info = [["数量", str(las_file_context.header.point_count)],
                ["X方向比例", str(las_file_context.header.x_scale)],
                ["Y方向比例", str(las_file_context.header.y_scale)],
                ["Z方向比例", str(las_file_context.header.z_scale)],
                ["X方向偏移", str(las_file_context.header.x_offset)],
                ["Y方向偏移", str(las_file_context.header.y_offset)],
                ["Z方向偏移", str(las_file_context.header.z_offset)],
                ["X坐标最大值", str(las_file_context.header.x_max)],
                ["X坐标最小值", str(las_file_context.header.x_min)],
                ["Y坐标最大值", str(las_file_context.header.y_max)],
                ["Y坐标最小值", str(las_file_context.header.y_min)],
                ["Z坐标最大值", str(las_file_context.header.z_max)],
                ["Z坐标最小值", str(las_file_context.header.z_min)]]
        with open(las_info_csv_file_path, "w", newline='') as csv_file:
            csv_file_writer = csv.writer(csv_file)
            csv_file_writer.writerows(las_info)
            
    towers_path = os.path.join(result_second_level_las_folder, "杆塔")
    if not os.path.exists(towers_path):
        os.mkdir(towers_path)
        
    towers_las_file_path = os.path.join(towers_path, "towers.las")
    if not os.path.exists(towers_las_file_path):
        unassigned_name_z_max_dict = {}
        for i in glob.glob(os.path.join(result_second_level_las_folder, "*.las")):
            las_file_context = laspy.read(i)
            if len(las_file_context.points) > 0:
                unassigned_name_z_max_dict[i] = las_file_context.z.max()
        towers_key = max(unassigned_name_z_max_dict, key=unassigned_name_z_max_dict.get)
        copyfile(towers_key, towers_las_file_path)
        
    towers_txt_file_path = os.path.join(towers_path, "towers.txt")
    if not os.path.exists(towers_txt_file_path):
        towers_las_file = laspy.read(towers_las_file_path)
        towers_txt_file = open(towers_txt_file_path, 'w')
        points = towers_las_file.points
        x_scale = towers_las_file.header.scale[0]
        y_scale = towers_las_file.header.scale[1]
        z_scale = towers_las_file.header.scale[2]
        x_offset = towers_las_file.header.offset[0]
        y_offset = towers_las_file.header.offset[1]
        z_offset = towers_las_file.header.offset[2]
        for point in points:
            x = point.X * x_scale + x_offset
            y = point.Y * y_scale + y_offset
            z = point.Z * z_scale + z_offset
            towers_txt_file.write(f"{x} {y} {z}\n")
        towers_txt_file.close()
        
    tower_asc_folder = os.path.join(towers_path , "临时文件")
    if not os.path.exists(tower_asc_folder):
        os.mkdir(tower_asc_folder)
    if not os.path.exists(os.path.join(tower_asc_folder, "segment_0.asc")):
        f = open(towers_txt_file_path )
        fr = f.readlines()
        x = []
        y = []
        z = []
        for line in fr:
            splits = line.split(' ')
            if len(splits) == 3:
                x.append(splits[0])
                y.append(splits[1])
                z.append(splits[2])
        f.close()
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        x = x.astype('float64')
        y = y.astype('float64')
        z = z.astype('float64')
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_y = y[sorted_indices]
        sorted_z = z[sorted_indices]
        sorted_points = np.column_stack((sorted_x, sorted_y, sorted_z))
        data = sorted_points
        dx = np.diff(data[:, 0])
        threshold = 50
        pos = np.where(dx > threshold)[0]
        segments = np.split(data, pos + 1)
        for i, seg in enumerate(segments):
            output_file_name = os.path.join(tower_asc_folder, f"segment_{i}.asc")
            np.savetxt(output_file_name, seg, delimiter=' ', fmt='%.16f', header='', comments='')
    tower_csv_folder = os.path.join(towers_path, "单个杆塔")
    if not os.path.exists(tower_csv_folder):
        os.mkdir(tower_csv_folder)
    if not os.path.exists(os.path.join(tower_csv_folder, "segment_0.csv")):
        files = glob.glob(os.path.join(tower_asc_folder, '*.asc'))
        count = 0
        for file in files:
            if ("segment" in file):
                data = np.genfromtxt(file, delimiter=' ', skip_header=0)
                num_columns = data.shape[1]
                constant_values = np.array([239, 239, 235])
                data = np.column_stack((data, np.tile(constant_values, (data.shape[0], 1))))
                data = data[data[:, 1].argsort()]
                dy = np.diff(data[:, 1])
                pos = np.where(dy > 50)[0]
                segments = np.split(data, pos + 1)
                for i, seg in enumerate(segments):
                    output_filename = os.path.join(tower_csv_folder, f"segment_{count}.csv")
                    np.savetxt(output_filename, seg, delimiter=',', fmt='%.16f', header='X,Y,Z,R,G,B', comments='')
                    count = count + 1
    tower_number = 10
    files = glob.glob(os.path.join(tower_csv_folder, '*.csv'))
    if tower_number < 0 and tower_number > len(files):
        print("杆塔编号超出范围")
    tower_line_number = os.path.join(towers_path, "towers.csv")
    if not os.path.exists(tower_line_number):
        info = ""
        for file in files:
            if ("segment" in file):
                file_path, full_name = os.path.split(file)
                file_name, ext = os.path.splitext(full_name)
                info = info + file_name
                if "segment_" + str(tower_number) == file_name:
                    print("单个杆塔点云模型的路径：" + file)
                data = np.genfromtxt(file, delimiter=',', skip_header=1)
                info = info + "," + str(np.mean(data[:, 0]))+ "," + str(np.mean(data[:, 1])) + "," + str(np.mean(data[:, 2])) + "\n"
        with open(tower_line_number,'w') as f:
            f.write(info)
    else:
        print("单个杆塔点云模型的路径：" + os.path.join(tower_csv_folder, "segment_" + str(tower_number) + ".csv"))
        os.startfile(tower_csv_folder)
    
