import tokenizers 

paths = ["../bucket/dataset.txt"]

# Initialize a tokenizer
bwpt = tokenizers.BertWordPieceTokenizer(vocab_file=None, add_special_tokens=True, unk_token='[UNK]', sep_token='[SEP]', cls_token='[CLS]', clean_text=True, wordpieces_prefix='##')

bwpt.train(files=paths, vocab_size=30000, min_frequency=3, limit_alphabet=1000, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]'] )

# Save files to disk
bwpt.save("../bucket/", "arabic")
