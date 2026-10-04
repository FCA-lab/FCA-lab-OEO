import matplotlib.pyplot as plt
import pandas as pd

result_path = "../data/result/result-for-random-comtext.csv"


def draw_with_density_change(obj_num, att_num):
    d = pd.read_csv(result_path)
    data = d.loc[d['obj_num'] == obj_num][d['att_num'] == att_num]
    x = data['desity']
    y_qian = data['qianting_time']
    y_ba = data['ba_time']
    plt.figure(dpi=600)
    A, = plt.plot(x, y_qian, "^:", label='the old algorithm')  # color="#fc8c5a",linestyle="solid",
    B, = plt.plot(x, y_ba, "o-", label='the new algorithm')  # color = '#4b74b2',linestyle="dotted",
    plt.title("obj num: {}, att num:{}".format(obj_num, att_num))
    plt.xlabel("desity")
    plt.ylabel("time (s)")
    legend = plt.legend(handles=[A, B])
    # for a, b in zip(x, y_qian):
    #     plt.text(a, b, b, ha='center', va='bottom')
    # for a, b in zip(x, y_ba):
    #     plt.text(a, b, b, ha='center', va='bottom')
    plt.savefig("../data/result/density/objNum{}-attNum{}-result.png".format(obj_num, att_num))


def draw_with_obj_change(density, att_num):
    d = pd.read_csv(result_path)
    data = d.loc[d['desity'] == density][d['att_num'] == att_num]
    x = data['obj_num']
    y_qian = data['qianting_time']
    y_ba = data['ba_time']
    plt.figure(dpi=600)
    A, = plt.plot(x, y_qian, "^:", label='the old algorithm')  # color="#fc8c5a",linestyle="solid",
    B, = plt.plot(x, y_ba, "o-", label='the new algorithm')  # color = '#4b74b2',linestyle="dotted",
    plt.title("density: {}, att num:{}".format(density, att_num))
    plt.xlabel("obj num")
    plt.ylabel("time (s)")
    legend = plt.legend(handles=[A, B])
    # for a, b in zip(x, y_qian):
    #     plt.text(a, b, b, ha='center', va='bottom')
    # for a, b in zip(x, y_ba):
    #     plt.text(a, b, b, ha='center', va='bottom')
    plt.savefig("../data/result/obj_num/density{}-attNum{}-result.png".format(density, att_num))


def draw_with_att_change(density, obj_num):
    d = pd.read_csv(result_path)
    data = d.loc[d['desity'] == density][d['obj_num'] == obj_num]
    x = data['att_num']
    y_qian = data['qianting_time']
    y_ba = data['ba_time']
    plt.figure(dpi=600)
    A, = plt.plot(x, y_qian, "^:", label='the old algorithm')  # color="#fc8c5a",linestyle="solid",
    B, = plt.plot(x, y_ba, "o-", label='the new algorithm')  # color = '#4b74b2',linestyle="dotted",
    plt.title("density: {}, obj num:{}".format(density, obj_num))
    plt.xlabel("att num")
    plt.ylabel("time (s)")
    legend = plt.legend(handles=[A, B])
    # for a, b in zip(x, y_qian):
    #     plt.text(a, b, b, ha='center', va='bottom')
    # for a, b in zip(x, y_ba):
    #     plt.text(a, b, b, ha='center', va='bottom')
    plt.savefig("../data/result/att_num/density{}-objNum{}-result.png".format(density, obj_num))


if __name__ == '__main__':
    for att_num in [10]:
        for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            draw_with_obj_change(density, att_num)

    # for obj_num in [10, 20, 30, 40, 50, 60]:
    #     for att_num in [10]:
    #         draw_with_density_change(obj_num, att_num)

    # for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #     for obj_num in [10]:
    #         draw_with_att_change(density, obj_num)
