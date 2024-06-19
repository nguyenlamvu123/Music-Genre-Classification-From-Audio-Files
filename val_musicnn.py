import os, json


genr_ori = os.path.join(os.sep, 'home', 'bhm-ai', 'music_classification', 'Data', 'genres_original')
valllltxt = os.path.dirname(__file__)


def readfile(file="uid.txt", mod="r", cont=None, jso: bool = False):
    if not mod in ("w", "a", ):
        assert os.path.isfile(file), str(file)
    if mod == "r":
        with open(file, encoding="utf-8") as file:
            lines: list = file.readlines()
        return lines
    elif mod == "_r":
        with open(file, encoding="utf-8") as file:
            contents = file.read() if not jso else json.load(file)
        return contents
    elif mod == "rb":
        with open(file, mod) as file:
            contents = file.read()
        return contents
    elif mod in ("w", "a", ):
        with open(file, mod, encoding="utf-8") as fil_e:
            if not jso:
                fil_e.write(cont)
            else:
                json.dump(cont, fil_e, indent=2, ensure_ascii=False)


def mscnn2txt():
    from musicnn.tagger import top_tags
    from tqdm import tqdm

    import warnings
    warnings.filterwarnings("ignore")

    # print('@@@@@@@@@@@', genr_ori)
    for mus_g in tqdm(os.listdir(genr_ori)):
        if mus_g in ('disco', 'hiphop', 'reggae', ):
            continue
        for mus_f in os.listdir(os.path.join(genr_ori, mus_g)):
            print('@@@@@@@@@@@@@@@@@@@@@', os.path.join(genr_ori, mus_g, mus_f))
            if mus_g in ('classical', 'country', 'metal', 'pop', 'rock', ):
                print(f'@@@@@@@@@@@@@@@@@@@@-MTT_musicnn----------{mus_g}')
                top_tags(os.path.join(genr_ori, mus_g, mus_f), model='MTT_musicnn', topN=3)
            if mus_g in ('blues', 'jazz', 'country', 'metal', 'pop', 'rock', ):
                print(f'@@@@@@@@@@@@@@@@@@@@-MSD_musicnn----------{mus_g}')
                top_tags(os.path.join(genr_ori, mus_g, mus_f), model='MSD_musicnn', topN=3)


def readresult():
    def initttt():
        return False, False, None, None, False

    MTTdict: dict = dict()
    MSDdict: dict = dict()
    for theloai in ('blues', 'jazz', 'country', 'metal', 'pop', 'rock', 'classical',):
        MTTdict[theloai] = {
            'correct': 0,
            'incorrect': [0, []],
            'total': 0,
        }
        MSDdict[theloai] = {
            'correct': 0,
            'incorrect': [0, []],
            'total': 0,
        }

    tenbai_ = '@@@@@@@@@@@@@@@@@@@@@ /home/bhm-ai/music_classification/Data/genres_original/'
    mtt_ = '@@@@@@@@@@@@@@@@@@@@-MTT_musicnn----------'
    msd_ = '@@@@@@@@@@@@@@@@@@@@-MSD_musicnn----------'
    top3tag = '[/home/bhm-ai/music_classification/'
    mttbool, msdbool, tenbai, theloai, dung = initttt()
    tagind: int = 3

    resu: list = readfile(file='/home/zaibachkhoa/Downloads/vallllllllllll.txt')
    for line in resu:
        if line.startswith(tenbai_):
            assert tagind == 3
            # assert theloai is None
            # assert tenbai is not None
            mttbool, msdbool, tenbai, theloai, dung = initttt()
            tenbai = line[len(tenbai_):].strip()
        elif line.startswith(mtt_):
            theloai = line[len(mtt_):].strip()
            assert theloai in MTTdict
            assert theloai in MSDdict
            assert tagind == 3
            mttbool = True
            msdbool = False
            tagind = 0
        elif line.startswith(msd_):
            theloai = line[len(msd_):].strip()
            assert theloai in MTTdict
            assert theloai in MSDdict
            assert tagind == 3
            mttbool = False
            msdbool = True
            tagind = 0
        elif line.startswith(top3tag):
            assert tenbai is not None
            assert tenbai in line
        elif line.startswith(' - '):
            tagind += 1
            if dung is False:
                assert theloai is not None
                if theloai in line:
                    dung = True
            if tagind == 3:
                if mttbool:
                    MTTdict[theloai]['total'] += 1
                    assert msdbool is False
                    if dung:
                        MTTdict[theloai]['correct'] += 1
                    else:
                        MTTdict[theloai]['incorrect'][0] += 1
                        MTTdict[theloai]['incorrect'][1].append(tenbai)
                if msdbool:
                    MSDdict[theloai]['total'] += 1
                    assert mttbool is False
                    if dung:
                        MSDdict[theloai]['correct'] += 1
                    else:
                        MSDdict[theloai]['incorrect'][0] += 1
                        MSDdict[theloai]['incorrect'][1].append(tenbai)
    cont: dict = {
        'MTT_musicnn': MTTdict,
        'MSD_musicnn': MSDdict,
    }
    readfile(file="result.json", mod="w", cont=cont, jso=True)


if __name__ == '__main__':
    readresult()
    mscnn2txt()
