#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, sys, unicodedata
import codecs
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, TextIO
import logging
from pypinyin import lazy_pinyin, Style


# gt_pinyin = lazy_pinyin(
#     gt_norm,
#     style=Style.TONE3,
#     tone_sandhi=True,
#     neutral_tone_with_five=True,
# )


remove_tag = True
spacelist = [' ', '\t', '\r', '\n']
puncts = [
    '!', ',', '?', '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』',
    '《', '》'
]


def characterize(string):
    res = []
    i = 0
    while i < len(string):
        char = string[i]
        if char in puncts:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        #https://unicodebook.readthedocs.io/unicode.html#unicode-categories
        if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist:  # space or not assigned
            i += 1
            continue
        if cat1 == 'Lo':  # letter-other
            res.append(char)
            i += 1
        else:
            # some input looks like: <unk><noise>, we want to separate it to two words.
            sep = ' '
            if char == '<': sep = '>'
            j = i + 1
            while j < len(string):
                c = string[j]
                if ord(c) >= 128 or (c in spacelist) or (c == sep):
                    break
                j += 1
            if j < len(string) and string[j] == '>':
                j += 1
            res.append(string[i:j])
            i = j
    return res


def stripoff_tags(x):
    if not x: return ''
    chars = []
    i = 0
    T = len(x)
    while i < T:
        if x[i] == '<':
            while i < T and x[i] != '>':
                i += 1
            i += 1
        else:
            chars.append(x[i])
            i += 1
    return ''.join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """ sentence, ignore_words are both in unicode
    """
    new_sentence = []
    for token in sentence:
        x = token
        if not cs:
            x = x.upper()
        if x in ignore_words:
            continue
        if remove_tag:
            x = stripoff_tags(x)
        if not x:
            continue
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence


class Calculator:

    def __init__(self):
        self.data = {}
        self.space = []
        self.cost = {}
        self.cost['cor'] = 0
        self.cost['sub'] = 1
        self.cost['del'] = 1
        self.cost['ins'] = 1

    def calculate(self, lab, rec):
        # Initialization
        lab.insert(0, '')
        rec.insert(0, '')
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element['dist'] = 0
                element['error'] = 'non'
            while len(row) < len(rec):
                row.append({'dist': 0, 'error': 'non'})
        for i in range(len(lab)):
            self.space[i][0]['dist'] = i
            self.space[i][0]['error'] = 'del'
        for j in range(len(rec)):
            self.space[0][j]['dist'] = j
            self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        # Computing edit distance
        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = sys.maxsize
                min_error = 'none'
                dist = self.space[i - 1][j]['dist'] + self.cost['del']
                error = 'del'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j - 1]['dist'] + self.cost['ins']
                error = 'ins'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['cor']
                    error = 'cor'
                else:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['sub']
                    error = 'sub'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]['dist'] = min_dist
                self.space[i][j]['error'] = min_error
        # Tracing back
        result = {
            'lab': [],
            'rec': [],
            'all': 0,
            'cor': 0,
            'sub': 0,
            'ins': 0,
            'del': 0
        }
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            if self.space[i][j]['error'] == 'cor':  # correct
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
                    result['all'] = result['all'] + 1
                    result['cor'] = result['cor'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'sub':  # substitution
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
                    result['all'] = result['all'] + 1
                    result['sub'] = result['sub'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'del':  # deletion
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
                    result['all'] = result['all'] + 1
                    result['del'] = result['del'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, "")
                i = i - 1
            elif self.space[i][j]['error'] == 'ins':  # insertion
                if len(rec[j]) > 0:
                    self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
                    result['ins'] = result['ins'] + 1
                result['lab'].insert(0, "")
                result['rec'].insert(0, rec[j])
                j = j - 1
            elif self.space[i][j]['error'] == 'non':  # starting point
                break
            else:  # shouldn't reach here
                print(
                    'this should not happen , i = {i} , j = {j} , error = {error}'
                    .format(i=i, j=j, error=self.space[i][j]['error']))
        return result

    def overall(self):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in self.data:
            result['all'] = result['all'] + self.data[token]['all']
            result['cor'] = result['cor'] + self.data[token]['cor']
            result['sub'] = result['sub'] + self.data[token]['sub']
            result['ins'] = result['ins'] + self.data[token]['ins']
            result['del'] = result['del'] + self.data[token]['del']
        return result

    def cluster(self, data):
        result = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        for token in data:
            if token in self.data:
                result['all'] = result['all'] + self.data[token]['all']
                result['cor'] = result['cor'] + self.data[token]['cor']
                result['sub'] = result['sub'] + self.data[token]['sub']
                result['ins'] = result['ins'] + self.data[token]['ins']
                result['del'] = result['del'] + self.data[token]['del']
        return result

    def keys(self):
        return list(self.data.keys())


def width(string):
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)


def default_cluster(word):
    unicode_names = [unicodedata.name(char) for char in word]
    for i in reversed(range(len(unicode_names))):
        if unicode_names[i].startswith('DIGIT'):  # 1
            unicode_names[i] = 'Number'  # 'DIGIT'
        elif (unicode_names[i].startswith('CJK UNIFIED IDEOGRAPH')
              or unicode_names[i].startswith('CJK COMPATIBILITY IDEOGRAPH')):
            # 明 / 郎
            unicode_names[i] = 'Mandarin'  # 'CJK IDEOGRAPH'
        elif (unicode_names[i].startswith('LATIN CAPITAL LETTER')
              or unicode_names[i].startswith('LATIN SMALL LETTER')):
            # A / a
            unicode_names[i] = 'English'  # 'LATIN LETTER'
        elif unicode_names[i].startswith('HIRAGANA LETTER'):  # は こ め
            unicode_names[i] = 'Japanese'  # 'GANA LETTER'
        elif (unicode_names[i].startswith('AMPERSAND')
              or unicode_names[i].startswith('APOSTROPHE')
              or unicode_names[i].startswith('COMMERCIAL AT')
              or unicode_names[i].startswith('DEGREE CELSIUS')
              or unicode_names[i].startswith('EQUALS SIGN')
              or unicode_names[i].startswith('FULL STOP')
              or unicode_names[i].startswith('HYPHEN-MINUS')
              or unicode_names[i].startswith('LOW LINE')
              or unicode_names[i].startswith('NUMBER SIGN')
              or unicode_names[i].startswith('PLUS SIGN')
              or unicode_names[i].startswith('SEMICOLON')):
            # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
            del unicode_names[i]
        else:
            return 'Other'
    if len(unicode_names) == 0:
        return 'Other'
    if len(unicode_names) == 1:
        return unicode_names[0]
    for i in range(len(unicode_names) - 1):
        if unicode_names[i] != unicode_names[i + 1]:
            return 'Other'
    return unicode_names[0]


def usage():
    print(
        "compute-wer.py : compute word error rate (WER) and align recognition results and references."
    )
    print(
        "         usage : python compute-wer.py [--cs={0,1}] [--cluster=foo] [--ig=ignore_file] [--char={0,1}] [--v={0,1}] [--padding-symbol={space,underline}] test.ref test.hyp > test.wer"
    )
    print()  # keep legacy usage for reference


# =============================================================================
# New argparse-based entry point handling a single combined recog file
# =============================================================================
WHITELIST = [
    ('地', '的'),
    ('的', '地'),
]
    
if __name__ == '__main__':
    # helper to parse boolean-like CLI arguments
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() not in ('false', '0', 'no', 'n')

    parser = argparse.ArgumentParser(
        description='Compute Word Error Rate (WER) using a combined recog file that contains both reference (ref= ...) and hypothesis (hyp= ...) lines.')

    # positional argument: combined file path
    parser.add_argument('recog_file',
                        help='Path to combined recog.txt file. Each line must look like "<utt_id>:\tref=..." or "<utt_id>:\thyp=..."')

    # options corresponding to the legacy switches
    parser.add_argument('--maxw', type=int, default=sys.maxsize,
                        help='Maximum number of words per printed line (legacy --maxw=)')
    parser.add_argument('--rt', type=str2bool, default=True,
                        help='Remove XML/HTML-style tags before scoring (legacy --rt=)')
    parser.add_argument('--cs', type=str2bool, default=False,
                        help='Case-sensitive evaluation (legacy --cs=)')
    parser.add_argument('--cluster', dest='cluster_file', default='',
                        help='Optional word-cluster definition file (legacy --cluster=)')
    parser.add_argument('--splitfile', default='',
                        help='Optional word split definition file (legacy --splitfile=)')
    parser.add_argument('--ig', default='',
                        help='Optional ignore-word list file (legacy --ig=)')
    parser.add_argument('--char', type=str2bool, default=True,
                        help='Tokenize at Unicode character level instead of word level (legacy --char=)')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbosity level 0/1/2 (legacy --v=)')
    parser.add_argument('--padding-symbol', choices=['space', 'underline'], default='space',
                        help='Padding symbol when printing alignment (legacy --padding-symbol=)')

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Map parsed arguments to variables expected by the original algorithm
    # ---------------------------------------------------------------------
    remove_tag = args.rt
    case_sensitive = args.cs
    cluster_file = args.cluster_file
    tochar = args.char
    verbose = args.verbose
    padding_symbol = ' ' if args.padding_symbol == 'space' else '_'
    max_words_per_line = args.maxw
    split = None
    ignore_words = set()

    # ---------------------------------------------------------------------
    # Load auxiliary resources (ignore list, split rules)
    # ---------------------------------------------------------------------
    if args.ig:
        with codecs.open(args.ig, 'r', 'utf-8') as fh:
            for line in fh:
                token = line.strip()
                if token:
                    ignore_words.add(token if case_sensitive else token.upper())

    if args.splitfile:
        split = {}
        with codecs.open(args.splitfile, 'r', 'utf-8') as fh:
            for line in fh:
                words = line.strip().split()
                if len(words) >= 2:
                    key = words[0]
                    values = words[1:]
                    if not case_sensitive:
                        key = key.upper()
                        values = [w.upper() for w in values]
                    split[key] = values

    # ---------------------------------------------------------------------
    # Parse combined recog file into reference/hypothesis dictionaries
    # ---------------------------------------------------------------------
    ref_set, rec_set, utter_order = {}, {}, []
    combined_line_re = re.compile(r'([^:]+):\s*(ref|hyp)\s*=\s*(.*)', re.IGNORECASE)

    with codecs.open(args.recog_file, 'r', 'utf-8') as fh:
        for raw_line in fh:
            line = raw_line.rstrip('\n')
            if not line:
                continue
            m = combined_line_re.match(line)
            if not m:
                continue  # skip malformed lines
            utt_id, kind, content = m.group(1).strip(), m.group(2).lower(), m.group(3).strip()

            tokens = characterize(content) if tochar else content.split()
            norm_tokens = normalize(tokens, ignore_words, case_sensitive, split)

            if utt_id not in utter_order:
                utter_order.append(utt_id)
            if kind == 'ref':
                ref_set[utt_id] = norm_tokens
            else:
                rec_set[utt_id] = norm_tokens

    # ---------------------------------------------------------------------
    # WER computation
    # ---------------------------------------------------------------------
    calculator = Calculator()
    default_clusters = {}
    default_words = {}

    # dictionaries for global error distribution
    subs = defaultdict(int)          # all substitution errors
    pinyin_subs = defaultdict(int)   # substitution errors with different pinyin
    ins = defaultdict(int)           # insertion errors
    dels = defaultdict(int)          # deletion errors

    for utt_id in utter_order:
        if utt_id not in ref_set or utt_id not in rec_set:
            continue  # incomplete pair
        lab, rec = ref_set[utt_id], rec_set[utt_id]

        if verbose:
            print(f"\nutt: {utt_id}")

        # accumulate default clusters for per-cluster WER later
        for word in rec + lab:
            if word not in default_words:
                cls_name = default_cluster(word)
                default_clusters.setdefault(cls_name, {})[word] = 1
                default_words[word] = cls_name

        result = calculator.calculate(lab, rec)

        # -----------------------------------------------------------------
        # Build arrow-style alignment and accumulate error statistics
        # -----------------------------------------------------------------
        arrow_tokens = []
        for ltok, rtok in zip(result['lab'], result['rec']):
            if ltok == rtok:
                arrow_tokens.append(ltok)
            elif ltok != '' and rtok != '':
                arrow_tokens.append(f'({ltok}->{rtok})')
                subs[(ltok, rtok)] += 1

                # determine if pinyin differs
                try:
                    p_ref = ''.join(lazy_pinyin(ltok, style=Style.TONE3, tone_sandhi=True, neutral_tone_with_five=True))
                    p_hyp = ''.join(lazy_pinyin(rtok, style=Style.TONE3, tone_sandhi=True, neutral_tone_with_five=True))
                except Exception:
                    p_ref, p_hyp = ltok, rtok  # fallback
                if p_ref != p_hyp:
                    if (ltok, rtok) not in WHITELIST:
                        pinyin_subs[(ltok, rtok)] += 1
            elif ltok != '' and rtok == '':
                arrow_tokens.append(f'({ltok}->*)')
                dels[ltok] += 1
            elif ltok == '' and rtok != '':
                arrow_tokens.append(f'(*->{rtok})')
                ins[rtok] += 1

        # ---------------------- per-utterance printout --------------------
        if verbose:
            wer = 0.0 if result['all'] == 0 else (result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
            print(f"WER: {wer:4.2f} % N={result['all']} C={result['cor']} S={result['sub']} D={result['del']} I={result['ins']}")
            print(' '.join(filter(lambda t: t != '', arrow_tokens)))

    # ---------------------------------------------------------------------
    # Overall WER statistics (unchanged logic)
    # ---------------------------------------------------------------------
    if verbose:
        print('=' * 75 + '\n')

    overall = calculator.overall()

    subs_total = overall['sub']
    pinyin_sub_total = sum(pinyin_subs.values())

    overall_wer_sub = 0.0 if overall['all'] == 0 else (overall['ins'] + subs_total + overall['del']) * 100.0 / overall['all']
    overall_wer_pinyin = 0.0 if overall['all'] == 0 else (overall['ins'] + pinyin_sub_total + overall['del']) * 100.0 / overall['all']

    if not verbose:
        print()

    # ---------------------- global error distribution --------------------
    if verbose:
        print()
        print('SUBSTITUTIONS: count ref -> hyp')
        for (ref_word, hyp_word), cnt in sorted(subs.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{cnt}   {ref_word} -> {hyp_word}")

        print()  # blank line
        print('PINYIN_SUBSTITUTIONS: count ref -> hyp')
        for (ref_word, hyp_word), cnt in sorted(pinyin_subs.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{cnt}   {ref_word} -> {hyp_word}")

        print()  # blank line
        print('DELETIONS: count ref')
        for ref_word, cnt in sorted(dels.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{cnt}   {ref_word}")

        print()  # blank line
        print('INSERTIONS: count hyp')
        for hyp_word, cnt in sorted(ins.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{cnt}   {hyp_word}")
        print()

    # ---------------------------------------------------------------------
    # Per-cluster WER (same as original implementation)
    # ---------------------------------------------------------------------
    print(f"Overall (substitution) -> {overall_wer_sub:4.2f} % N={overall['all']} C={overall['cor']} S={subs_total} D={overall['del']} I={overall['ins']}")
    print(f"Overall (pinyin_substitution) -> {overall_wer_pinyin:4.2f} % N={overall['all']} C={overall['cor']} S={pinyin_sub_total} D={overall['del']} I={overall['ins']}")
    print('=' * 75)
    if verbose:
        for cluster_id in default_clusters:
            stats = calculator.cluster(list(default_clusters[cluster_id].keys()))

            # compute substitution counts for this cluster
            sub_cnt_cluster = stats['sub']
            p_sub_cnt = sum(cnt for (ref_word, _), cnt in pinyin_subs.items() if default_words.get(ref_word) == cluster_id)

            c_wer_sub = 0.0 if stats['all'] == 0 else (stats['ins'] + sub_cnt_cluster + stats['del']) * 100.0 / stats['all']
            c_wer_pinyin = 0.0 if stats['all'] == 0 else (stats['ins'] + p_sub_cnt + stats['del']) * 100.0 / stats['all']

            print(f"{cluster_id} (substitution) -> {c_wer_sub:4.2f} % N={stats['all']} C={stats['cor']} S={sub_cnt_cluster} D={stats['del']} I={stats['ins']}")
            print(f"{cluster_id} (pinyin_substitution) -> {c_wer_pinyin:4.2f} % N={stats['all']} C={stats['cor']} S={p_sub_cnt} D={stats['del']} I={stats['ins']}")

        # legacy explicit cluster file support
        if cluster_file:
            cluster_id, cluster_terms = '', []
            with open(cluster_file, 'r', encoding='utf-8') as fh:
                for token in fh.read().split():
                    # end of cluster like </Keyword>
                    if token.startswith('</') and token.endswith('>') and token.lstrip('</').rstrip('>') == cluster_id:
                        stats = calculator.cluster(cluster_terms)
                        c_wer = 0.0 if stats['all'] == 0 else (stats['ins'] + stats['sub'] + stats['del']) * 100.0 / stats['all']
                        print(f"{cluster_id} -> {c_wer:4.2f} % N={stats['all']} C={stats['cor']} S={stats['sub']} D={stats['del']} I={stats['ins']}")
                        cluster_id, cluster_terms = '', []
                    # begin of cluster like <Keyword>
                    elif token.startswith('<') and token.endswith('>') and not cluster_id:
                        cluster_id = token.lstrip('<').rstrip('>')
                    # regular term within a cluster
                    else:
                        cluster_terms.append(token)
