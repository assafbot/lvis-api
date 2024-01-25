from lvis import LVISEval
import time

if __name__ == "__main__":
    # result and val files for 100 randomly sampled images.
    # ANNOTATION_PATH = "./data/lvis_val_100.json"
    # RESULT_PATH = "./data/lvis_results_100.json"
    ANNOTATION_PATH = '/home/talshah/Downloads/lvis_v1_val.json'
    RESULT_PATH = '/home/talshah/Downloads/predictions.json'
    ANN_TYPE = 'bbox'
    a = time.time()
    lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH, ANN_TYPE)
    lvis_eval.run()
    print('AP50: {}'.format(lvis_eval.results['AP50']))
    print('final: {}'.format(time.time()/60 - a/60))