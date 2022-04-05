import matplotlib.pyplot as plt
from PIL import Image
#img = Image.open('data/stinkbug.png')
import numpy as np
import copy
import os
import cv2

def img_add(front, back, stx, sty):
    w, h = front.shape
    res = copy.deepcopy(back)
    res[stx:stx+w, sty:sty+h] = res[stx:stx+w, sty:sty+h]+front
    return res


def gen_img_data():
    return gen_cat_data_simple()


def video2frame(video_src_path, frame_save_path, frame_width, frame_height, interval, start, end):
    """
    将视频按固定间隔读取写入图片
    :param video_src_path: 视频存放路径
    :param formats:　包含的所有视频格式
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    videos = os.listdir(video_src_path)
    for each_video in videos:
        print("reding：", each_video)  # 我的是Python3.6

        each_video_name = each_video[:-4]
        if not os.path.isdir(frame_save_path + each_video_name):
            os.mkdir(frame_save_path + each_video_name)
        each_video_save_full_path = os.path.join(frame_save_path, each_video_name) + "_"

        each_video_full_path = os.path.join(video_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        frame_index = 0
        frame_count = 0
        if cap.isOpened():
            success = True
        else:
            success = False
            print("Reading fail!")

        while (success):
            success, frame = cap.read()
            print("---> 正在读取第%d帧:" % frame_index, success)  # 我的是Python3.6

            if frame_index>start and frame_index % interval == 0 and success:  # 如路径下有多个视频文件时视频最后一帧报错因此条件语句中加and success
                resize_frame = frame#cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
                cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, resize_frame)
                frame_count += 1
                print(each_video_save_full_path + "%d.jpg" % frame_count)
                print('saving')

            frame_index += 1
            if frame_index>end:
                break
            #cap.release()  # 这行要缩一下、原博客会报错


def process_cat_data():
    videos_src_path = "D:\TOOLS\jjdown\Download\监控拍下两只流浪猫来我家蹭吃蹭喝，一只吃另一只望风！"
    # video_formats = [".MP4", ".MOV"]          我的数据集都是.mp4所以不需要进行分类判断
    frames_save_path = "video\cats"
    width = 720
    height = 480
    start=1200
    end=1800
    time_interval = 10
    video2frame(videos_src_path, frames_save_path, width, height, time_interval, start, end)

def process_cat_data_xb():
    videos_src_path = "video\catscat"
    # video_formats = [".MP4", ".MOV"]          我的数据集都是.mp4所以不需要进行分类判断
    frames_save_path = "video\catscat"
    width = 720
    height = 480
    start = 0
    end = 100000
    time_interval = 10
    video2frame(videos_src_path, frames_save_path, width, height, time_interval, start, end)

def save_image(image,addr,num):
    cv2.saveimg(addr+str(num)+".jpg", image)

def gen_cat_data_simple():
    #names = ['c1.jpg','c2.jpg','c3.jpg','c4.jpg']#,'c5.jpg',]
    names = ["cat_"+str(i)+".jpg" for i in range(20)]
    res = []
    for name in names:
        img = Image.open('video/cats/'+name)
        img = np.array(img)
        cat = np.mean(img, axis=2)
        cat = cat[0:400,200:700]
        res.append(cat)

        print(cat.shape)
    #print(len(res))
    return np.stack(res)


def gen_football_data():
    img = Image.open('video/football.png')
    img = img.resize((40,40), Image.ANTIALIAS)
    img = np.array(img)
    ftb = np.mean(img, axis=2)
    print(ftb.shape)

    img = Image.open('video/grass2.jpg')
    #img = img.resize((100,100), Image.ANTIALIAS)
    img = np.array(img)
    #bg = img[]
    #bg = np.mean(img, axis=2)
    bg = np.mean(img[:200,:200,:], axis=2)
    print(bg.shape)
    compose = np.stack([img_add(ftb, bg, i, i) for i in [20, 30, 60, 120, 150]])

    return compose

def imgsshow(compose):
    for c in compose:
        plt.imshow(c)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    #imgsshow(gen_img_data())
    process_cat_data_xb()