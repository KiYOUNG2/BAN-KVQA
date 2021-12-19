import numpy as np
import os
from .constants import Config as C

# Load classes
CLASSES = ['__background__']
with open(os.path.join(C.DETECTOR_DATA_PATH, C.OBJECT_VOCAB_FILE)) as f:
    for object in f.readlines():
        CLASSES.append(object.split(',')[0].lower().strip())

# Load attributes
ATTRIBUTES = ['__no_attribute__']
with open(os.path.join(C.DETECTOR_DATA_PATH, C.ATTR_VOCAB_FILE)) as f:
    for att in f.readlines():
        ATTRIBUTES.append(att.split(',')[0].lower().strip())

CLASSES = np.array(CLASSES)
ATTRIBUTES = np.array(ATTRIBUTES)


# rule 2 : 색, rule 3 : 모양, rule 4 : 무늬
ATTSET_RULE2 = set(ATTRIBUTES[
    np.array([1,   2,   5,   7,   8,   9,  10,  11,  12,  13,  15,  18,  24,
        30,  31,  32,  36,  73,  92, 118, 164, 196, 205, 221, 235, 254,
       255, 266, 267, 277, 285, 294, 295, 303, 317, 369, 399])])

ATTSET_RULE3 = set(ATTRIBUTES[
    np.array([4,  16,  42, 55,  61,  83,  94,  97, 102, 112, 116, 119, 120, 121,
       124, 126, 132, 151, 162, 171, 187, 190, 194, 206, 211, 212, 216,
       228, 256, 268, 271, 278, 283, 292, 299, 302, 304, 329, 349, 360,
       371, 379, 381, 385, 387, 391, 392, 393, 398])])

ATTSET_RULE4 = set(ATTRIBUTES[
    np.array([6,  17,  52, 131, 191, 281, 332])])

COUNT_NOUN = np.array([
    '가지', '개', '개가', '개이', '개지', '구', '구간', '권', '그루', '단', '매', '매가', '명', '명의',
    '명이', '마리', '분할', '사람', '색', '알', '이', '인분', '자루', '장',
    '조각', '종', '좌석', '줄', '층', '채', '칸', '켤레', '팀', '포기'])

# Rules
RULE1_SNT = [
         '이것 은 무엇',
         '이 건 무엇',
         '이것 이 무엇',
         '저 건 무엇',
         '저것 은 무엇',
         '무엇 이 ㄴ가요',
         '무엇 이 ㄴ지',
         '무엇 이 보이',
         '무엇 이 진열',
         '무엇 인 이 ㄴ가요',
         '이 건 무엇',
         '이것 은 무엇',
         '이것 의 이름 은 무엇',
         '이것 이 무엇',
         '저 건 무엇',
         '저것 은 무엇',
         '화면 에 무엇 이',
         '무엇 이 ㄴ가',
         '이 건 뭐 야',
         '저 건 뭐 야',
         '뭐 야',
         '뭐 니',
         '이것 이 뭐 니',
         '이것 이 니',
         '이 건 뭐 니',
         '이것 이 뭐 야',
        '뭐 가 보이 어',
        '뭐 가 보이 나요',
        '뭐 가 아 보이 나 아요',
        '화연 에 무엇 이 보이 나요 ?',
        '사진 에 무엇 이 보이 나요 ?',
        '앞 에 무엇 이 보이 나요 ?',
        ]

RULE2_SNT = [
        '무슨 색',
        '무슨 색깔',
        '어떤 색깔',
        '이 건 무슨 색 이',
        '이거 무슨 색깔 이 야',
        '이 건 어떤 색 이',
        '이 건 어떤 색깔 이',
        '이거 무슨 색 이',
        '이거 무슨 색깔 이',
        '이것 은 무슨 색 이',
        '이것 은 무슨 색깔 이',
        '이것 이 무슨 색 이',
        '이것 이 무슨 색깔 이',
        '색깔 알려주 어',
        ]

RULE3_SNT = [
    '무슨 모양',
    '어떤 모양',
    '무슨 모양 이',
    '어떤 모양 이',
    '어떻 ㄴ 모양 이',
    '이거 무슨 모양 이',
    '이것 은 무슨 모양',
    '이것 이 무슨 모양'
        ]

RULE4_SNT = [
    '무슨 무늬 가',
    '무슨 무늬 이',
    '무슨 무늬 야',
    '어떤 무늬 가',
    '어떤 무늬 이',
    '어떤 무늬 야',
    '무늬 가 뭐 야 ?',
    '무늬 알려주 어',
    ]

RULE6_SNT = [
    '가 듣 어 있 나요 ?',
    '가 보이 나요 ?',
    '가 있 나요 ?',
    '가 보이 니 ?',
    '가 보이 어 ?',
    '가 있 니 ?',
    '가 있 니 ?',
    '이 ㄴ가요 ?',
    '이 듣 어 있 나요 ?',
    '이 보이 어 ?',
    '이 보이 나요 ?',
    '이 보이 ㅂ니까 ?',
    '이 안 보이 어 ?',
    '이 안 보이 나요 ?',
    '이 안 보이 ㅂ니까 ?',
    '이 맞 니 ?',
    '이 맞 아 ?',
    '이 야 ?',
    '이 있 나요 ?',
    '이 있 습니까 ?',
    '이 있 어 ?',
    '있 나요 ?',
    '가 있 어 ?'
    '있 어 ?'
    '가 아니 야 ?',
    '가 없 나요 ?',
    '가 없 니 ?',
    '없 나요 ?',
    '없 어 ?',
    '이 아니 야 ?',
    '이 없 나요 ?',
    '이 없 습니까 ?',
    '이 없 어 ?'
]

RULE2_TOKEN = ['색','색깔']
RULE3_TOKEN = '모양'
RULE4_TOKEN = '무늬'
RULE5_TOKEN = '몇'