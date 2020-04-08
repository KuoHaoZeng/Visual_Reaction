import sys
import numpy as np
import jsonlines

def eval_general(learn_result):
    e, m, h, num_hits = [], [], [], []
    v, a, p = [], [], []
    num = 0
    for xx in learn_result:
        if 'ball_pos_est' in xx.keys():
            l = (((np.array(xx['ball_pos_est']) - np.array(xx['ball_pos'])) ** 2).sum(1) ** 0.5).mean()
            p.append(l)
        if 'ball_vel_est' in xx.keys():
            l = (((np.array(xx['ball_vel_est']) - np.array(xx['ball_vel'])) ** 2).sum(1) ** 0.5).mean()
            v.append(l)
        if 'ball_acc_est' in xx.keys():
            l = (((np.array(xx['ball_acc_est']) - np.array(xx['ball_acc'])) ** 2).sum(1) ** 0.5).mean()
            a.append(l)
        hits = max(xx['num_SimObj_hits']) + max(xx['num_Structure_hits']) - max(xx['num_Floor_hits'])
        if hits == 0:
            e.append(max(xx['success']))
        elif hits < 3: # 2
            m.append(max(xx['success']))
        else:
            h.append(max(xx['success']))
        num += 1
        num_hits.append(hits)
    e, m, h = np.array(e), np.array(m), np.array(h)
    v, a, p = np.array(v), np.array(a), np.array(p)
    total_success = e.sum() + m.sum() + h.sum()
    print("-------")
    print("total performance over {} trajectories: {:.2f}%".format(num, 100 * total_success / num))
    print("position L2: {:.3f}  ± {:.3f}".format(np.mean(p), np.std(p)))
    print("velocity L2: {:.3f}  ± {:.3f}".format(np.mean(v), np.std(v)))
    print("accleration L2: {:.3f}  ± {:.3f}".format(np.mean(a), np.std(a)))
    print("-------")
    print("\t#easy: {} ({:.2f}%)\t#medium: {} ({:.2f}%)\t#hard: {} ({:.2f}%)".format(len(e), len(e)*100/num,
                                                                                  len(m), len(m)*100/num,
                                                                                  len(h), len(h)*100/num))
    print("Acc.\t\t{:.2f}%\t\t\t{:.2f}%\t\t\t{:.2f}%".format(e.mean() * 100, m.mean() * 100, h.mean() * 100))

def eval_category(learn_result):
    output = {}
    for xx in learn_result:
        obj = xx['object_name'][0]
        result = max(xx['success'])
        if obj in output.keys():
            output[obj].append(result)
        else:
            output[obj] = [result]
    print("-------")
    txt1 = "\t"
    txt2 = "Acc.\t"
    cnt = 0
    for k, v in output.items():
        txt1+="{:10}\t".format(k)
        txt2+="{:5.2f}%  \t".format(100 * np.array(v).mean())
        cnt += 1
        if cnt == 5:
            print(txt1)
            print(txt2)
            cnt = 0
            txt1 = "\t"
            txt2 = "Acc.\t"
    print("-------")

if __name__ == "__main__":
    base_dir = sys.argv[1]
    dataset = sys.argv[2]

    learn_result = jsonlines.Reader(open('{}/{}_log.jsonl'.format(base_dir, dataset)))
    eval_general(learn_result)

    learn_result = jsonlines.Reader(open('{}/{}_log.jsonl'.format(base_dir, dataset)))
    eval_category(learn_result)
