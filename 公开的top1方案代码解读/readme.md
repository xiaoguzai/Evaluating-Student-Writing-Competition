# Kaggle Feedback Prize-Evaluate Student Writing代码解读

最近参加了kaggle的Evaluate Student Writing比赛，受限于自身水平有限，没能够取得一个很好的名次，所以这里学习一下榜一大哥的代码，以便于今后有更大的进步。

[比赛地址](https://www.kaggle.com/c/feedback-prize-2021)

[大佬的解法](https://www.kaggle.com/c/feedback-prize-2021)

## 1.代码的解读

首先确定标签内容

```python
discourse_type = ['Claim','Evidence', 'Position','Concluding Statement','Lead','Counterclaim','Rebuttal']
b_discourse_type = ['I-'+i for i in discourse_type]
i_discourse_type = ['B-'+i for i in discourse_type]
```

接下来定义标签的序号

```python
args.labels_to_ids = {k:v for v,k in enumerate(['O']+b_discourse_type+i_discourse_type)}
args.ids_to_labels = {k:v for v,k in args.labels_to_ids.items()}
```

得到标签的内容

```
args.labels_to_ids = {'O': 0, 'I-Claim': 1, 'I-Evidence': 2, 'I-Position': 3, 
'I-Concluding Statement': 4, 'I-Lead': 5, 'I-Counterclaim': 6, 'I-Rebuttal': 7, 
'B-Claim': 8, 'B-Evidence': 9, 'B-Position': 10, 'B-Concluding Statement': 11, 
'B-Lead': 12, 'B-Counterclaim': 13, 'B-Rebuttal': 14}
args.ids_to_labels = {0: 'O', 1: 'I-Claim', 2: 'I-Evidence', 3: 'I-Position', 
4: 'I-Concluding Statement', 5: 'I-Lead', 6: 'I-Counterclaim', 7: 'I-Rebuttal', 
8: 'B-Claim', 9: 'B-Evidence', 10: 'B-Position', 11: 'B-Concluding Statement', 
12: 'B-Lead', 13: 'B-Counterclaim', 14: 'B-Rebuttal'}
```

使用get_feat函数得到相应的数据

```python
train_feat = get_feat(train_df,tokenizer,args,'train_feat'+args.key_string)
test_feat = get_feat(test_df,tokenizer,args,'test_feat'+args.key_string)
```

**这里的get_feat将train和test进行切词预训练，得到切词结果并将切词结果保存下来**

```python
from joblib import Parallel,delayed
def get_feat(df, tokenizer, args, data_key):
    data_path = args.cache_path + 'feat_{}.pkl'.format(data_key)
    #data_path = /home/xiaoguzai/数据/evaluate-student-writing/cache/feat_train_featlongformer_v2_15class_adv_fold0.pkl
    #这里使用pkl文件保存切词结果
    if os.path.exists(data_path) & (args.load_feat):
        data = pickle.load(open(data_path,'+rb'))
    else:
        num_jobs = 16
        data = []
        train_ids = df["id"].unique()
        #注意这里的unique()很细节
        train_ids_splits = np.array_split(train_ids, num_jobs)
        #np.array_split:不均等分割
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(get_feat_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
        )
        #get_feat_helper函数在上面，获取对应的切分结果
        for result in results:
            data.extend(result)
            r"""
            data输入的内容为{'id':idx,'text':text,'input_ids':input_ids,\
            'attention_mask':attention_mask,'token_label':token_label,\
            'offset_mapping':offset_mapping}
            """
        data = pd.DataFrame(sorted(data,key=lambda x:len(x['input_ids'])))
        pickle.dump(data,open(data_path,'+wb'))
    return data
```

其中多线程切分词语

```python
results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(get_feat_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
        )
```

这里调用了get_feat_helper函数内容，我们查看get_feat_helper函数

```python
def get_feat_helper(args, tokenizer, df, train_ids):
    training_samples = []
    for idx in train_ids:
        filename = args.text_path + idx + ".txt"
        with open(filename, "r") as f:
            text = f.read().rstrip()
        input_ids, attention_mask, token_label, offset_mapping = \
            encode(text,tokenizer,df[df['id']==idx],args.labels_to_ids)
        training_samples.append({'id':idx,'text':text,'input_ids':input_ids,
                             'attention_mask':attention_mask,'token_label':token_label,
                            'offset_mapping':offset_mapping})
    return training_samples
```

可以看出就是将不同的文本内容进行切词并将结果放入对应的list

接下来调用相应的参数，这里我们需要查看一下Collate函数的调用

```python
args.train_batch_size = 1
train_params = {'batch_size': args.train_batch_size,
                    'shuffle': True, 'num_workers': 2, 'pin_memory':True,
                    'collate_fn':Collate(args.max_length)
                    }
args.valid_batch_size = 1
test_params = {'batch_size': args.valid_batch_size,
               'shuffle': False, 'num_workers': 2,'pin_memory':True,
               'collate_fn':Collate(4096)
              }
```

这里的Collate函数是从每一个batch之中切分出对应内容并转化为对应的tensor，具体代码如下：

```python
class Collate:
    def __init__(self, model_length=None,max_length=None,padding_side='right',padding_dict={}):
        self.model_length = model_length
        self.max_length   = max_length
        self.padding_side = padding_side
        self.padding_dict = padding_dict
    
    def __call__(self, batch):
        .............
```

这里的padding_side = right,padding_dict = {}

这里的Collate类刚刚完成初始化，我们先不深入它的call函数进行阅读

**构建DataLoader有一个很巧妙的地方，就是将参数以dict字典类型传入函数之中**

```python
train_loader = DataLoader(dataset(train_feat), **train_params)
test_loader = DataLoader(dataset(test_feat), **test_params)
```

接下来定义参数args.num_train_steps

```python
args.num_train_steps = len(train_feat)*args.epochs/args.train_batch_size
```

这里len(train_feat)应该为整个数据的长度，而不是整个数据的长度除以batch_size，所以乘上args.epochs之后需要除以args.train_batch_size。

接下来我们看模型的定义

```python
model = TextModel(args.model_name, num_labels=len(args.labels_to_ids))
model = torch.nn.DataParallel(model)
model.to(args.device)
```

## 2.模型的构建

**这里的模型需要重点看一下**

**1.首先需要注意的是模型的结构**

```python
class TextModel(nn.Module):
    def __init__(self,model_name=None,num_labels=1):
        super(TextModel,self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name) # 768
        self.drop_out = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        #config.hidden_size = 768
        self.output = nn.Linear(config.hidden_size,num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        if 'gpt' in self.model.name_or_path:
            emb = self.model(input_ids)[0]
        else:
            emb = self.model(input_ids,attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5
		.......
```

这里五个dropout然后求平均来源于一篇论文，证明dropout再平均之后的效果比原来模型的效果好(注意预测的时候模型需要调整一下)

**2.接下来需要注意的是pytorch之中softmax激活函数的使用，pytorch之中softmax激活函数不能跟CrossEntropy损失函数一起使用，否则会造成训练的结果都一样**

**3.最后是mask掩码的调用，计算损失函数注意mask的掩码**

```python
def get_loss(self, outputs, targets, attention_mask):
    loss_fct = nn.CrossEntropyLoss()
    active_loss = attention_mask.reshape(-1) == 1
    active_logits = outputs.reshape(-1, outputs.shape[-1])
    true_labels = targets.reshape(-1)
    idxs = np.where(active_loss.cpu().numpy() == 1)[0]
    #!!!这里的mask内容非常关键
    active_logits = active_logits[idxs]
    true_labels = true_labels[idxs].to(torch.long)
    loss = loss_fct(active_logits, true_labels)
    return loss
```

## 3.训练函数的解读

```python
model,test_pred = train(model,train_loader,test_loader,test_feat,test_df,args,model_path)
torch.save(model.state_dict(), "model.bin")
```

### CosineAnnealingWarmupRestarts学习函数的解读

```python
optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr)
scheduler = CosineAnnealingWarmupRestarts(optimizer = optimizer,
                                          first_cycle_steps = args.num_train_steps, cycle_mult = 1,
                                          max_lr = args.max_lr, min_lr = args.min_lr, warmup_steps = args.num_train_steps * 0.2,
                                          gamma = 1.,last_epoch = -1
                                         )
```

这里first_cycle_steps为整个的steps

```python
first_cycle_steps = args.num_train_steps
```

而取first_cycle_steps中的0.2作为warmup_steps

max_lr设定le-5，min_lr设定1e-6，进入到学习率的变换之中阅读函数

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau,_LRScheduler
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
```

这里面初始化的各类参数为：

```
self.first_cycle_steps = 24954.0
self.cycle_mult = 1
self.base_max_lr = 1e-05
self.max_lr = 1e-05
self.min_lr = 1e-06
self.warmup_steps = 4990.8
self.gamma = 1.0
self.cur_cycle_steps = 24954.0
self.step_in_cycle = 0
```

接下来查看函数之中调用的公式

```python
def get_lr(self):
    if self.step_in_cycle == -1:
        return self.base_lrs
    elif self.step_in_cycle < self.warmup_steps:
        return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
    else:
        return [base_lr + (self.max_lr - base_lr)
                 * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                 / (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs]
```

1.小于warmup_steps时候的学习率

```python
return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
```

对应公式为

$\eta_{min}+\frac{(\eta_{max}-\eta_{min})*self.step_in_cycle}{self.warmup_steps}$

其中$\eta$代表学习率，$\eta_{min}$代表最小学习率，$\eta__{max}$代表最大学习率

2.大于warmup_steps时候的学习率

```python
return [base_lr + (self.max_lr - base_lr)
        * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps))) / 2
        for base_lr in self.base_lrs]
```

对应的公式为

$\eta_{min}+\frac{1}{2}*(\eta_{max}-\eta_{min})*(1+cos(\pi*\frac{self.step\_in\_cycle-self.warmup\_steps}{self.cur\_cycle\_steps-self.warmup\_steps}))$

**这里需要注意的是，新型的CosineAnnealingWarmupRestarts是以batch_size为单位的，每训练一个batch_size之后就需要更新一次，而不是像有一些以epoch为单位的学习率，每一个epoch才进行一次更新**

运行step函数的内容

```python
def step(self, epoch=None):
    if epoch is None:
        epoch = self.last_epoch + 1
        self.step_in_cycle = self.step_in_cycle + 1
        if self.step_in_cycle >= self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
            self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
	...........
```

首先这里的epoch一直为None，所以这里一直运行的是第一个if，后续的else没有再被运行，所以这里暂时不探究else函数之中运行的内容

```python
if epoch is None:
    epoch = self.last_epoch + 1
    self.step_in_cycle = self.step_in_cycle + 1
    if self.step_in_cycle >= self.cur_cycle_steps:
        self.cycle += 1
        self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
        self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
```

这里cycle_mult = 1，因此这里self.cur_cycle_steps保持不变，而self.step_in_cycle = self.step_in_cycle-self.cur_cycle_step更新一波，保证warmup之后每个step()的学习率保持一致

### EarlyStopping类的内容解读

定义EarlyStopping类型

```python
es = EarlyStopping(patience=4,max_epoch=...)
```

EarlyStopping的内部结构就不细讲了，通俗地来说就是如果patience个epoch都没有能够更新最高分，则模型停止训练。

### AWP对抗网络内容解读

```python
scaler = torch.cuda.amp.GradScaler()
awp = AWP(model,\
         optimizer,\
         adv_lr=args.adv_lr,\
         adv_eps=args.adv_eps,
         start_epoch=args.num_train_steps/args.epochs,
         scaler=scaler)
```

这里初始化的参数为

```
args.adv_lr = 0.0000
args.adv_eps = 0.001
```

接下来阅读其中更新参数的代码

```python
if f1_score > 0.64:
    awq.attack_backward(input_ids,labels,attention_mask,step)
```



首先attack攻击对应的梯度信息

```python
def attack_backward(self, x, y, attention_mask,epoch):
    if (self.adv_lr == 0) or (epoch < self.start_epoch):
        return Nonep
    self._save() 
    for i in range(self.adv_step):
        self._attack_step() 
        with torch.cuda.amp.autocast():
            adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
            adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()
            self._restore()
```



### collate函数的调用

接下来进入到提取数据的环节

```python
for idx,batch in enumerate(train_loader):
    ......
```

这里会调用collate函数对数据进行调用

```python
def __call__(self, batch):
    output = dict()
    output["input_ids"] = [sample["input_ids"] for sample in batch]
    output["attention_mask"] = [sample["attention_mask"] for sample in batch]
    output["labels"] = [sample["labels"] for sample in batch]

    # calculate max token length of this batch
    batch_length = None
    if self.model_length is not None:
        batch_length = self.model_length
        else:
            batch_length = max([len(ids) for ids in output["input_ids"]])
            if self.max_length is not None:
                batch_length = min(batch_length,self.max_length)

                #batch_length = 1024
                #self.padding_dict = {},self.padding_side = right
                for i in range(len(output["input_ids"])):
                    output_fill = feat_padding(output["input_ids"][i], output["attention_mask"][i],output["labels"][i],
                                               batch_length,self.padding_dict,padding_side=self.padding_side)
                    output["input_ids"][i],output["attention_mask"][i], output["labels"][i] = output_fill

                    # convert to tensors
                    output["input_ids"]      = torch.stack(output["input_ids"])
                    output["attention_mask"] = torch.stack(output["attention_mask"])
                    output["labels"]         = torch.stack(output["labels"])
                    #torch.stack()：把多个2维的张量凑成一个三维的张量，多个三维的张量凑成一个四维的张量
                    #也就是在增加新的维度进行堆叠
                    return output
```

这里调用数据就是平平无奇的进行切分并填充，让它构成相应的tensor，具体的填充过程在feat_padding之中。

**很巧妙的地方在于，feat_padding之中安排了不同的截取长度的方法，非常值得学习**

截取长度的函数方法

```python
def feat_padding(input_ids,attention_mask,token_label,batch_length,padding_dict,padding_side):
    random_seed = None
    if padding_side == 'right':
        random_seed = 0
    elif padding_side == 'left':
        random_seed = 1
    else:
        random_seed = np.random.rand()
                
    mask_index = attention_mask.nonzero().reshape(-1)
    input_ids      = input_ids.index_select(0,mask_index)
    token_label    = token_label.index_select(0,mask_index)
    attention_mask = attention_mask.index_select(0,mask_index)
    ids_length = len(input_ids)
```

当然目前作者还是使用的右边的padding内容

```python
output_fill = feat_padding(output["input_ids"][i],output["attention_mask"][i],\
                          output["labels"][i],batch_length,self.padding_dict,\
                          padding_side=self.padding_side)
output["input_ids"][i],output["attention_mask"][i], output["labels"][i] = output_fill
```

**除此之外，在训练之中我认为非常值得学习的地方在于当分数达到一定值的时候进行对抗训练，这点非常巧妙**

```python
if f1_score > 0.64:
    awp.attack_backward(input_ids,labels,attention_mask,step)
```

### 大于某一个零界值进行对抗

这是在模型训练之中值得尝试的一个方向

# 评判指标的解读

对于模型的评判指标以及得分的解读，还需要进一步阅读代码获得