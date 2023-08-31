import os
import json

import torch
from PIL import Image
from torchvision import transforms

# from model import resnet34
# from model import resnet34,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d
# from model_change_downsaple import resnet50
# from model_change_downsaple_al_lwt import resnet50
# from model_change_downsaple_all import resnet50
# from model_change_downsaple_att_test import resnet50
# from model_change_downsaple_attention import resnet50
# from model_SE_Resnet50 import SEResNet
# from model_50 import resnet50
# from model_shufflenet import shufflenet_v2_x0_5
# from model_v2_Mobile import MobileNetV2
# from model_densenet import densenet121
# from model_googlenet import GoogLeNet
# from model_50 import resnet50
from model_SE_Resnet50_att_avg_dwtlayer4 import SEResNet


def main():
    # 更改部分1
    pthpath = "1100/model_second_resnet50_1ceng.pth"
    # model = SEResNet(730)
    model = resnet50(730)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    # 指向需要遍历预测的图像文件夹
    # imgs_root = "../data/imgs"
    imgs_root = "../data_set/lsn_data/test_20_safe"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i)
                     for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    total_num = len(img_path_list)

    # read class_indict
    json_path = './class_indices_flower.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    lables = dict((val, key) for key, val in class_indict.items())

    # create model
    # model = resnet50(num_classes=730).to(device)
    model.to(device)
    # model = SEResNet(730).to(device)

    # load model weights

    weights_path = "../logs/"+pthpath
    assert os.path.exists(
        weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction1
    model.eval()
    batch_size = 32  # 每次预测时将多少张图片打包成一个batch
    top1, top2, top3, top4, top5 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size+1):
            img_list = []
            lable_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(
                    img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path).convert("RGB")
                # linux
                # lable = img_path.split("/")[-1].split("_")[0]
                # windows
                lable = img_path.split("\\")[-1].split("_")[0]
                lable = lables[lable]
                lable_list.append(lable)
                img = data_transform(img)
                img_list.append(img)

            # batch img  将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)

            # probs 概率从大到小排序，classes 预测标签
            # descending为alse，升序，为True，降序
            probs, classes = torch.sort(predict, descending=True)
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                # print(lable_list[idx])
                # print(cla[:5])
                if int(lable_list[idx]) in cla[:1]:
                    top1 = top1+1
                if int(lable_list[idx]) in cla[:2]:
                    top2 = top2+1
                if int(lable_list[idx]) in cla[:3]:
                    top3 = top3+1
                if int(lable_list[idx]) in cla[:4]:
                    top4 = top4+1
                if int(lable_list[idx]) in cla[:5]:
                    top5 = top5+1
                print("image: {}  class1: {}  prob1: {:.3} class2: {}  prob2: {:.3} "
                      "class3: {}  prob3: {:.3} class4: {}  prob4: {:.3} "
                      "class5: {}  prob5: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                        class_indict[str(
                                                            cla[0].numpy())], pro[0].numpy(),
                                                        class_indict[str(
                                                            cla[1].numpy())], pro[1].numpy(),
                                                        class_indict[str(
                                                            cla[2].numpy())], pro[2].numpy(),
                                                        class_indict[str(
                                                            cla[3].numpy())], pro[3].numpy(),
                                                        class_indict[str(cla[4].numpy())], pro[4].numpy()))
            print(
                "top1:{}  top2:{}  top3:{}  top4:{}  top5:{}".format(top1, top2, top3, top4, top5))
            print(
                "top1:{:.5}  top2:{:.5}  top3:{:.5}  top4:{:.5}  top5:{:.5}".format(top1 / total_num, top2 / total_num,
                                                                                    top3 / total_num, top4 / total_num,
                                                                                    top5 / total_num))
            print(
                "{:.5} {:.5} {:.5} {:.5} {:.5}".format(top1 / total_num, top2 / total_num,
                                                       top3 / total_num, top4 / total_num,
                                                       top5 / total_num))
            print("第{}轮结束！".format(ids))
        print("测试集图片数：", total_num)
        print("测试结束！")


if __name__ == '__main__':
    main()
