def gettext():
    txt = open("../data/wikibio/raw/train/train.sent", "r", errors='ignore').read()
    # table = open("/data/wikibio/raw/train/train.sent","r",errors='ignore').read()
    # for ch in '!"#$&()*+,-./:;<=>?@[\\]^_{|}·~‘’':
    #     txt = txt.replace(ch, "")
    return txt


if __name__ == '__main__':

    txt = gettext()
    words = txt.split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    items = list(counts.items())

    for i in [1,5,10,20,30,40,50,60,70,100]:
        new_items = []
        total_count = 0
        thread_count = 0
        for item in items:
            total_count += item[1]
            if item[1] > i:
                thread_count += item[1]
                new_items.append(item)
        print(total_count)
        print('{}_词汇表长度：{}，占比：{:.4f}'.format(i, len(new_items), thread_count/total_count))

    # items.sort(key=lambda x:x[1],reverse=True)

    # for i in range(20):
    #     word, count = items[i]
    #     print("{0:<10}{1:>5}".format(word,count))
