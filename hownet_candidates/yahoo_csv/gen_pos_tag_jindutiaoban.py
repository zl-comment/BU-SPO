import pickle
from tqdm import tqdm
from nltk.tag import StanfordPOSTagger

# 之前的注释代码
'''
with open('./all_seqs.pkl', 'rb') as fh:
    train, valid, test = pickle.load(fh)
with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)
dict = {w: i for (w, i) in tokenizer.word_index.items()}
inv_dict = {i: w for (w, i) in dict.items()}
word_candidate = {}
trains = [t[1:-1] for t in train['s2']]
'''
#生成词性标注的！
# 加载数据集
with open('aux_files/small_dataset_50000.pkl', 'rb') as fp:
    dataset = pickle.load(fp)

# 配置Stanford POS tagger
jar = '/home/cyh/ZLCODE_SPO/data_set/stanford-postagger-full-2020-11-17/stanford-postagger.jar'
model = '/home/cyh/ZLCODE_SPO/data_set/stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

# 准备训练集和测试集文本
train_text = [[dataset.inv_full_dict[t] for t in tt] for tt in dataset.train_seqs]
test_text = [[dataset.inv_full_dict[t] for t in tt] for tt in dataset.test_seqs]

# 初始化词性标注列表
all_pos_tags = []
test_pos_tags = []

# 处理训练集文本，并显示进度条
for text in tqdm(train_text, desc="Processing train texts"):
    pos_tags = pos_tagger.tag(text)
    all_pos_tags.append(pos_tags)

# 处理测试集文本，并显示进度条
for text in tqdm(test_text, desc="Processing test texts"):
    pos_tags = pos_tagger.tag(text)
    test_pos_tags.append(pos_tags)

# 保存训练集的词性标注结果
with open('pos_tags.pkl', 'wb') as f:
    pickle.dump(all_pos_tags, f)

# 保存测试集的词性标注结果
with open('pos_tags_test.pkl', 'wb') as f:
    pickle.dump(test_pos_tags, f)