import sys
import os
import pdb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

sys.path.insert(0, ".")
from liver_detector import AI_LiverDetect


def test(gpu_id=0):

    consumer = AI_LiverDetect()
    consumer.init(gpu_id)
    consumer.prepare()
    print('prepare done')
    subjects = [
        'data_example/1166948/1.2.86.76547135.7.6592994.20191106092400/1.2.840.113619.2.334.3.604658089.159.1572251051.61_0,data_example/1166948/1.2.86.76547135.7.6592994.20191106092400/1.2.840.113619.2.334.3.604658089.159.1572251051.182_0,data_example/1166948/1.2.86.76547135.7.6592994.20191106092400/1.2.840.113619.2.334.3.604658089.159.1572251051.182_1',
        'data_example/1163646/1.2.86.76547135.7.7338893.20200914102400/1.2.840.113619.2.334.3.1712603141.510.1599782153.635_0,data_example/1163646/1.2.86.76547135.7.7338893.20200914102400/1.2.840.113619.2.334.3.1712603141.510.1599782153.688_0,data_example/1163646/1.2.86.76547135.7.7338893.20200914102400/1.2.840.113619.2.334.3.1712603141.510.1599782153.688_1',
        'data_example/1163603/1.2.86.76547135.7.7554448.20201121154500/1.3.12.2.1107.5.1.4.73133.30000020112100014221600044227_0,data_example/1163603/1.2.86.76547135.7.7554448.20201121154500/1.3.12.2.1107.5.1.4.73133.30000020112100014221600044305_0,data_example/1163603/1.2.86.76547135.7.7554448.20201121154500/1.3.12.2.1107.5.1.4.73133.30000020112100014221600044341_0',
        'data_example/1163153/1.2.86.76547135.7.7605160.20201207153217/1.2.840.113619.2.334.3.604658089.485.1606662092.311_0,data_example/1163153/1.2.86.76547135.7.7605160.20201207153217/1.2.840.113619.2.334.3.604658089.485.1606662092.411_0,data_example/1163153/1.2.86.76547135.7.7605160.20201207153217/1.2.840.113619.2.334.3.604658089.485.1606662092.411_1'
    ]
    for idx, subject in enumerate(subjects):
        uid = subject.split("/")[-3]
        task_info = {"ID": 656,
                     "INPUT_DIR": subject
                     }
        ret, msg = consumer.main_course(task_info)
        # print(s.getvalue())
        if ret is not None:
           open("liver_{}.json".format(uid), "w").write(ret)

    
if __name__=="__main__":
    if len(sys.argv) > 1:
        test(gpu_id=int(sys.argv[1]))
    else:
        test()
