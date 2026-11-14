from __future__ import annotations
from typing import Optional
import regex
#import tiktoken
import numpy as np
import datetime
from multiprocessing import Pool
import os
import traceback
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
import json
class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int], vocab_size: int) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self._pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self._mergeable_ranks = mergeable_ranks

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(self._pat_str)
        self._special_tokens = ["<|begin_of_text|>",  # Marks the beginning of a text sequence.
                                "<|end_of_text|>",  # Marks the end of a text sequence.
                                "<|reserved_special_token_0|>",  # Reserved for future use.                                
                                "<|reserved_special_token_1|>",  # Reserved for future use.
                                "<|finetune_right_pad_id|>",
                                "<|reserved_special_token_2|>",  # Reserved for future use.                                
                                "<|start_header_id|>",  # Indicates the start of a header ID.
                                "<|end_header_id|>",  # Indicates the end of a header ID.
                                "<|eom_id|>",  
                                "<|eot_id|>",  # Marks the end of a turn (in a conversational context).
                                "<|python_tag|>",
                                "<|reserved_special_token_3|>",  # Reserved for future use.
                                "<|reserved_special_token_4|>",  # Reserved for future use.                                
                               ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 8)]  # A large set of tokens reserved for future use.
        self._special_tokens = {token: (vocab_size + i) for i, token in enumerate(self._special_tokens)}

    def encode(self, text: str, visualise: str | None = "colour") -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.strip().encode("utf-8")
            word_tokens = bpe_encode(self._mergeable_ranks, word_bytes, visualise=visualise)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b" ".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str, mergeable_ranks: dict):
        """Train a BPE tokeniser on some data!"""
        _mergeable_ranks = bpe_train(data=training_data, vocab_size=vocab_size, pat_str=pat_str, 
                                     mergeable_ranks=mergeable_ranks, visualise='visualise')
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=_mergeable_ranks, vocab_size=vocab_size)


def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: str | None = "colour"
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if visualise:
            if visualise in ["colour", "color"]:
                visualise_tokens(parts)
            elif visualise == "simple":
                print(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    if visualise:
        print()

    tokens = [mergeable_ranks[part] for part in parts]
    return tokens


def bpe_train(
    data: str, vocab_size: int, pat_str: str, mergeable_ranks: dict, visualise: str | None = "colour"
) -> dict[bytes, int]:


    if mergeable_ranks:        
        if vocab_size < len(mergeable_ranks):
            raise ValueError("vocab_size must be greater lenght mergeable_ranks")

        ranks = mergeable_ranks
    else:
        # First, add tokens for each individual byte value
        if vocab_size < 2**8:
            raise ValueError("vocab_size must be at least 256, so we can encode all bytes")

        ranks = {}
        for i in range(2**8):
            ranks[bytes([i])] = i
    
    print(datetime.datetime.now(), flush=True)    
    #workers = 46
    chuncks = int(len(data)/50000)+1
    if chuncks > 20:
        workers = 20
    else:
        workers = chuncks
    stats = dict()
    words = dict()    
    with Pool(processes = workers) as pool:                
        for w, s in pool.imap_unordered(_prepare_initial_words_stats, 
                                          [(i, data[a[0]:a[-1]+1]) for i, a in enumerate(np.array_split(range(len(data)), chuncks)) if len(a)>0]):
            if isinstance(w, Exception) | isinstance(s, Exception):
                print("Got exception: {}".format(w))
            else:
                #print('prepare_initial_words_stats', len(s), len(w), flush=True)
                for key, value in s.items():                    
                    if key in stats:
                        freq = stats[key][0] + value[0]
                        ws = stats[key][1].copy()
                        ws.extend(value[1])
                    else:
                        freq = value[0]
                        ws = value[1]
                    stats[key] = (freq, ws)                    
                words.update(w)
                
    
    while len(ranks) < vocab_size:
        print(datetime.datetime.now(), flush=True)

        if len(stats) == 0:
            return ranks

        most_common_pair= max(stats, key=lambda k: stats[k][0])#max(stats, key = stats.get)        
        most_common_pair_words = stats[most_common_pair][1]
        chuncks = int(len(most_common_pair_words)/10000)+1
        if chuncks > 78:
            workers = 78
        else:
            workers = chuncks
        print(len(ranks), len(stats), most_common_pair, stats[most_common_pair][0], len(most_common_pair_words), chuncks, workers, flush=True)
        
        token_bytes = most_common_pair[0] + most_common_pair[1]        
        if ranks.get(token_bytes) is None: 
            token = len(ranks)
            # Add the new token!
            ranks[token_bytes] = token

        with Pool(processes = workers, initializer=init, initargs=(list(stats.keys()), words, most_common_pair)) as pool:                
            for s, w, r in pool.imap_unordered(merge_word, 
                                              [(i, most_common_pair_words[a[0]:a[-1]+1]) for i, a in enumerate(np.array_split(range(len(most_common_pair_words)), chuncks)) if len(a)>0]):
                if isinstance(w, Exception) | isinstance(s, Exception):
                    print("Got exception: {}".format(w))
                else:
                    #print('merge_word', len(s), len(w), len(r), flush=True)
                    for key, value in s.items():                        
                        if key in stats:
                            freq = stats[key][0] + value[0]
                            ws = stats[key][1].copy()
                            ws.extend(value[1])
                        else:
                            freq = value[0]
                            ws = value[1]
                        stats[key] = (freq, ws)
                    
                    for key, value in w.items():
                        words[key] = value
                    for key, value in r.items():
                        stats[key] = (stats[key][0] + value[0], list(set(stats[key][1]) - set(value[1])))
                        
                        if stats[key][0] == 0:
                            stats.pop(key)
                        
        print(datetime.datetime.now(), flush=True)
        
        # See the intermediate merges play out!
        if visualise:
            
            print(f"The current most common pair is {most_common_pair[0]} + {most_common_pair[1]}")
            print(f"So we made {token_bytes} our {len(ranks)}th token")
            if visualise in ["colour", "color"]:
                print("Now the first hundred words in our training data look like:")
                #visualise_tokens([token for word in list(zip(*data[:100]))[0] for token in word])
                visualise_tokens([token for k in list(words)[:100] for token in words[k][0]])
            elif visualise == "simple":
                print("Now the first twenty words in our training data look like:")
                #for word in list(zip(*data[:20]))[0]:
                for k in list(words)[:100]:
                    print(words[k][0])
            print("", flush=True)

    return ranks

def print_file(file, v):
    caminho = '/parallel/'+ file
    f = open(caminho, "a+")
    f.write(v + '->' + str(datetime.datetime.now()))
    f.write("\r\n")
    f.close()


def prepare_initial_words_stats(v, work_item, words, stats):        
    word: list[bytes] = [bytes([b]) for b in v[0].strip().encode("utf-8")]    
    
    pairs = zip(word[:-1], word[1:])
    words[v[0]] = (word, v[1])
    for pair in pairs:
        if pair in stats:
            ws = stats[pair][1]
            if v[0] not in ws:
                ws.append(v[0])
            stats[pair] = (stats[pair][0] + v[1], ws)
        else:
            stats[pair] = (v[1], [v[0]])
    
    return words, stats

def _prepare_initial_words_stats(arr):    
    work_item = arr[0]
    words = dict()
    stats = dict()
    try:
        print_file(str(work_item) + '.txt', 'INICIO_prepare_initial_words_stats')
        for v in arr[1]:
            words, stats = prepare_initial_words_stats(v, work_item, words, stats)
        print_file(str(work_item) + '.txt', 'FIM_prepare_initial_words_stats')
        return words, stats
    except Exception as ex:
        return Exception("Err on item {}".format(work_item)
                         + os.linesep + traceback.format_exc()), None

def init(s, w, m):    
    global stats
    stats = s
    global words
    words = w
    global most_common_pair
    most_common_pair = m
    
#def merge_word(most_common_pair, stats, words, most_common_pair_words):
def merge_word(most_common_pair_words):    
    words_new = dict()
    stats_new = dict()
    stats_remove = dict()
    for _w in most_common_pair_words[1]:
        word = words[_w] 
        w = word[0]
        
        new_word = []
        i=0
        j=0
        while i < len(w) - 1:    
            pair = (w[i], w[i + 1])
            if pair in stats_remove:
                ws = stats_remove[pair][1].copy()
                if _w not in ws:
                    ws.append(_w)
                stats_remove[pair] = (stats_remove[pair][0] - word[1], ws)
            else:
                stats_remove[pair] = (-word[1], [_w])

            if i == j:
                if pair == most_common_pair:
                    # We found our pair! Merge it
                    new_word.append(most_common_pair[0] + most_common_pair[1])
                    j += 2
                else:
                    new_word.append(w[j])
                    j += 1    
            i +=1
        if j == len(w) - 1:
            new_word.append(w[j])
        
        words_new[_w] = (new_word, word[1])
        pairs = zip(new_word[:-1], new_word[1:])
        for pair in pairs:
            if pair not in stats:
                if pair in stats_new:
                    ws = stats_new[pair][1].copy()
                    if _w not in ws:
                        ws.append(_w)
                    stats_new[pair] = (stats_new[pair][0] + word[1], ws)
                else:
                    stats_new[pair] = (word[1], [_w])
            
            if pair in stats_remove:
                ws = stats_remove[pair][1].copy()
                if _w in ws:
                    ws.remove(_w)
                stats_remove[pair] = (stats_remove[pair][0] + word[1], ws)
                if stats_remove[pair][0] == 0:
                    stats_remove.pop(pair)
            
    return stats_new, words_new, stats_remove

def merge_word_old(most_common_pair, stats, words):
    for _w in stats[most_common_pair][1]:
        word = words[_w] 
        w = word[0]
        new_word = []
        i=0
        j=0
        while i < len(w) - 1:    
            pair = (w[i], w[i + 1])
            ws = stats[pair][1].copy()
            ws.remove(_w)
            
            stats[pair] = (stats[pair][0] - word[1], ws)
            
            if stats[pair][0] == 0:
                stats.pop(pair)

            if i == j:
                if pair == most_common_pair:
                    # We found our pair! Merge it
                    new_word.append(most_common_pair[0] + most_common_pair[1])
                    j += 2
                else:
                    new_word.append(w[j])
                    j += 1    
            i +=1
        if j == len(w) - 1:
            new_word.append(w[j])
        
        words[_w] = (new_word, word[1])
        pairs = zip(new_word[:-1], new_word[1:])
        for pair in pairs:
            if pair in stats:
                ws = stats[pair][1].copy()
                ws.append(_w)
                stats[pair] = (stats[pair][0] + word[1], ws)
            else:
                stats[pair] = (word[1], [_w])        
    return stats, words
    
        
def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # visualise the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")

def get_llama3_pattern():
    return (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

def get_gpt2_pattern():
    return  (
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

def train_simple_encoding():    
    with open(__file__) as f:
        data = f.read()

    enc = SimpleBytePairEncoding.train(data, vocab_size=600, pat_str=get_llama3_pattern())

    print("This is the sequence of merges performed in order to encode 'hello world':")
    tokens = enc.encode("hello world")
    assert enc.decode(tokens) == "hello world"
    assert enc.decode_bytes(tokens) == b"hello world"
    assert enc.decode_tokens_bytes(tokens) == [b"hello", b" world"]

    return enc

def token_bytes_to_string(b):
    byte_encoder = bytes_to_unicode()

    return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

def generate_vocab_and_merges(encoder):
    mergeable_ranks = encoder._mergeable_ranks
    
    merges = []
    vocab = {}
    for token, rank in mergeable_ranks.items():
        vocab[token_bytes_to_string(token)] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(merged) == 2

        merges.append(' '.join(map(token_bytes_to_string, merged)))

    # Also add special tokens
    vocab.update(encoder._special_tokens)

    return vocab, merges

# Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960
def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

#def convert_tiktoken(model_name, output_dir=None):
#    encoder = tiktoken.encoding_for_model(model_name)
#    
#    convert_tiktoken(encoder, model_name, output_dir=None)
    
def convert_tiktoken(encoder, model_name, output_dir=None):
    if output_dir is None:
        output_dir = model_name

    MODEL_INFO = {
            'llama3_2': {
                'tokenizer_class': 'PreTrainedTokenizerFast',
                'model_max_length': 131072,
            },
        }

    vocab, merges = generate_vocab_and_merges(encoder)

    added_tokens = [
        {
          "id": id,
          "content": content,
          "single_word": False,
          "lstrip": False,
          "rstrip": False,
          "normalized": False,
          "special": True,
        }
        for content, id in encoder._special_tokens.items()
    ]

    # https://huggingface.co/Xenova/gpt2/raw/main/tokenizer_config.json
    tokenizer_config_template = {
    "add_prefix_space": False,
    "bos_token": "<|begin_of_text|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|endoftext|>",
    "unk_token": "<|endoftext|>",
    }
    
    special_tokens_map = {
          "bos_token": "<|begin_of_text|>",
          "eos_token": "<|endoftext|>",
          "unk_token": "<|endoftext|>",
        }

    os.makedirs(output_dir, exist_ok=True)

    post_processor = None
    if MODEL_INFO[model_name]['tokenizer_class'] in ('GPT3_5Tokenizer', 'GPT4Tokenizer'):
        pre_tokenizer = {
          "type": "Sequence",
          "pretokenizers": [
            {
              "type": "Split",
              "pattern": {
                "Regex": encoder._pat_str#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
              },
              "behavior": "Removed",
              "invert": True,
            },
            {
              "type": "ByteLevel",
              "add_prefix_space": False,
              "trim_offsets": True,
              "use_regex": False,
            }
          ]
        }
    elif MODEL_INFO[model_name]['tokenizer_class'] in ('PreTrainedTokenizerFast'):
        special_tokens_map = {
          "bos_token": {
            "content": "<|begin_of_text|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
          },
          "eos_token": {
            "content": "<|end_of_text|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False
          }
        }
        
        added_tokens_decoder = dict({
                                    str(id): 
                                      { 
                                        "content": content,                                        
                                        "lstrip": False,                                        
                                        "normalized": False,
                                        "rstrip": False,
                                        "single_word": False,
                                        "special": True,
                                      }
                                      for content, id in encoder._special_tokens.items()
                                })
        
        tokenizer_config_template = {
          #"add_prefix_space": False,
          "bos_token": "<|begin_of_text|>",
          "clean_up_tokenization_spaces": True,
          "eos_token": "<|end_of_text|>",
          #"unk_token": "<|end_of_text|>",
          "model_input_names": [
            "input_ids",
            "attention_mask"
          ],
          "added_tokens_decoder": added_tokens_decoder
        }
        
        pre_tokenizer = {
          "type": "Sequence",
          "pretokenizers": [
            {
              "type": "Split",
              "pattern": {
                "Regex": encoder._pat_str#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
              },
              "behavior": "Isolated",
              "invert": False,
            },
            {
              "type": "ByteLevel",
              "add_prefix_space": False,
              "trim_offsets": True,
              "use_regex": False,
            }
          ]
        }
        
        post_processor = {
          "type": "Sequence",
          "processors": [
            {
              "type": "ByteLevel",
              "add_prefix_space": True,
              "trim_offsets": False,
              "use_regex": True
            },
            {
              "type": "TemplateProcessing",
              "single": [
                {
                  "SpecialToken": {
                    "id": "<|begin_of_text|>",
                    "type_id": 0
                  }
                },
                {
                  "Sequence": {
                    "id": "A",
                    "type_id": 0
                  }
                }
              ],
              "pair": [
                {
                  "SpecialToken": {
                    "id": "<|begin_of_text|>",
                    "type_id": 0
                  }
                },
                {
                  "Sequence": {
                    "id": "A",
                    "type_id": 0
                  }
                },
                {
                  "SpecialToken": {
                    "id": "<|begin_of_text|>",
                    "type_id": 1
                  }
                },
                {
                  "Sequence": {
                    "id": "B",
                    "type_id": 1
                  }
                }
              ],
              "special_tokens": {
                "<|begin_of_text|>": {
                  "id": "<|begin_of_text|>",
                  "ids": [
                    encoder._special_tokens["<|begin_of_text|>"]
                  ],
                  "tokens": [
                    "<|begin_of_text|>"
                  ]
                }
              }
            }
          ]
        }
        
    else:
        pre_tokenizer = {
          "type": "ByteLevel",
          "add_prefix_space": False,
          "trim_offsets": True,
          "use_regex": True,
        }
        
    
    tokenizer_config_template.update(MODEL_INFO[model_name]) # Adds `model_max_length` and `tokenizer_class`
    tokenizer_config_template = dict(sorted(tokenizer_config_template.items(), key=lambda x: x[0]))

    # https://huggingface.co/Xenova/gpt2/raw/main/tokenizer.json
    tokenizer_template = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": {
          "type": "ByteLevel",
          "add_prefix_space": True,
          "trim_offsets": True,
          "use_regex": True,
        },
        "model": {
          "type": "BPE",
          "dropout": None,
          "unk_token": None,
          "continuing_subword_prefix": None,
          "end_of_word_suffix": None,
          "fuse_unk": False,
          "byte_fallback": False,
          "ignore_merges": True,
          "vocab": vocab,
          "merges": merges,
        },
    }
    
    # Save to files
    with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump(vocab, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'tokenizer.json'), 'w', encoding='utf-8') as fp:
        json.dump(tokenizer_template, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as fp:
        json.dump(tokenizer_config_template, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as fp:
        json.dump(special_tokens_map, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'merges.txt'), 'w', encoding='utf-8') as fp:
        fp.write('#version: 0.2\n')
        fp.write('\n'.join(merges))