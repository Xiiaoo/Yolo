import json
import os
 
 
def labelme2yolo_seg(class_name, json_dir, labels_dir):
    """
        此函数用来将labelme软件标注好的json格式转换为yolov_seg中使用的txt格式
        :param json_dir: labelme标注好的*.json文件所在文件夹
        :param labels_dir: 转换好后的*.txt保存文件夹
        :param class_name: 数据集中的类别标签
        :return:
    """
    list_labels = []  # 存放json文件的列表
 
    # 0.创建保存转换结果的文件夹
    os.makedirs(labels_dir, exist_ok=False)
 
    # 1.获取目录下所有的labelme标注好的Json文件，存入列表中
    for files in os.listdir(json_dir):  # 遍历json文件夹下的所有json文件
        file = os.path.join(json_dir, files)  # 获取一个json文件
        list_labels.append(file)  # 将json文件名加入到列表中
 
    for labels in list_labels:  # 遍历所有json文件
        with open(labels, "r") as f:
            file_in = json.load(f)
            shapes = file_in["shapes"]
            print(labels)
 
        txt_filename = os.path.basename(labels).replace(".json", ".txt")
        txt_path = os.path.join(labels_dir, txt_filename)  # 使用labels_dir变量指定保存路径
 
        with open(txt_path, "w+") as file_handle:
            for shape in shapes:
                line_content = []  # 初始化一个空列表来存储每个形状的坐标信息
                line_content.append(str(class_name.index(shape['label'])))  # 添加类别索引
                # 添加坐标信息
                for point in shape["points"]:
                    x = point[0] / file_in["imageWidth"]
                    y = point[1] / file_in["imageHeight"]
                    line_content.append(str(x))
                    line_content.append(str(y))
                # 使用空格连接列表中的所有元素，并写入文件
                file_handle.write(" ".join(line_content) + "\n")
                
                
# labelme2yolo_seg(
#     ['x', 'z', 'c'],
#     'D:\\JDSignUp\\0611\\img_labels',
#     'D:\\JDSignUp\\0611\\img_labels_yolo'
# )




import os
import random
import shutil

# 原始文件夹路径
images_dir = 'D:\\JDSignUp\\0611\\img_new'
labels_dir = 'D:\\JDSignUp\\0611\\img_labels_yolo'

# 目标文件夹路径
train_image_dir = 'D:\\JDSignUp\\0611\\imgs\\train'
val_image_dir = 'D:\\JDSignUp\\0611\\imgs\\val'
train_label_dir = 'D:\\JDSignUp\\0611\\labels\\train'
val_label_dir = 'D:\\JDSignUp\\0611\\labels\\val'

# 创建输出文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取图像文件列表
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 打乱并划分
random.shuffle(image_files)
split_idx = int(0.9 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# 拷贝函数
def copy_split(file_list, img_src, lbl_src, img_dst, lbl_dst):
    for fname in file_list:
        # 拷贝图像
        shutil.copy(os.path.join(img_src, fname), os.path.join(img_dst, fname))
        # 拷贝标签
        label_name = os.path.splitext(fname)[0] + '.txt'
        label_src_path = os.path.join(lbl_src, label_name)
        label_dst_path = os.path.join(lbl_dst, label_name)
        if os.path.exists(label_src_path):
            shutil.copy(label_src_path, label_dst_path)
        else:
            print(f"⚠️ 缺少标签: {label_src_path}")

# 执行拷贝
copy_split(train_files, images_dir, labels_dir, train_image_dir, train_label_dir)
copy_split(val_files, images_dir, labels_dir, val_image_dir, val_label_dir)

print("✅ 划分完成！图像和标签已按 9:1 存放到 images/train, images/val 和 labels/train, labels/val。")