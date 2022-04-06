import os
import yaml


# this script works only if all names sorted by clases, for example, img1 and img2 - first class, img3 and img4 - second class
def main():
    classes_numbers = [20, 20, 20, 20, 20]
    here = os.path.abspath(os.path.dirname(__file__))
    abspath = os.path.join(here, 'imgs' + os.sep)
    names_lst = sorted(os.listdir(abspath))

    start_val = 0
    res_dct = dict()
    for i, val in enumerate(classes_numbers):
        for name in names_lst[start_val:start_val + val]:
            res_dct[abspath + name] = i
        start_val += val

    with open(here + os.sep + 'dataset.yaml', 'w+') as out_f:
        yaml.dump(res_dct, out_f, default_flow_style=False)


if __name__ == '__main__':
    main()
