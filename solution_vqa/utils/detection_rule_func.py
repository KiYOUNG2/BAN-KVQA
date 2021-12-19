from typing import Union, List, Tuple

from configs.detection_rule import (
    CLASSES,
    ATTSET_RULE2,
    ATTSET_RULE3,
    ATTSET_RULE4,
    COUNT_NOUN,

    RULE1_SNT,
    RULE2_SNT,
    RULE3_SNT,
    RULE4_SNT,
    RULE6_SNT,
    RULE2_TOKEN,
    RULE3_TOKEN,
    RULE4_TOKEN,
    RULE5_TOKEN,
)

def rule_base_answer(
    snt:str,
    detected_objs:List[str],
    detected_atts:List[str],
    detected_dist_centers:List[Tuple],
    tagger):
    """Rule-based answer function
    Detected Objects를 이용하여 질문 유형에 따라 답변
    Args:
        snt (str): [description]
        tagger ([type]): [description]
    """
    # 답변 불가 
    if len(snt) == 0:
        return 'unanswerable'
    
    # tokenizing
    snt_morphs = tagger.morphs(snt)
    answer = 'unanswerable'

    # Rule 1 : 무엇(e.g. 이것은 무엇인가요?)
    if qcheck_rule_snt(snt_morphs, RULE1_SNT):
        print("rule 1")
        answer = rule_uncertain_obj_what(
                            objs=detected_objs,
                            atts=detected_atts,
                            dist_centers=detected_dist_centers,
                            )
        return answer
    
    # Rule 2-1 : 색, 색깔 - 주어가 없거나 지시대명사
    elif qcheck_rule_snt(snt_morphs, RULE2_SNT):
        print("rule 2")
        answer = rule_uncertain_obj(
                                atts=detected_atts,
                                dist_centers=detected_dist_centers,
                                att_set=ATTSET_RULE2,
                                )
        return answer

    # Rule 2-2 : 색, 색깔 - 주어가 있는 경우
    elif qcheck_rule_token(snt_morphs, RULE2_TOKEN):
        print("rule 2-token")
        answer = rule_certain_obj(
                    snt=snt, 
                    objs=detected_objs,
                    atts=detected_atts,
                    att_set=ATTSET_RULE2,
                    rule_tokens=RULE2_TOKEN,
                    tagger=tagger)

        return answer

    # rule 3-1 : 모양 - 주어가 없거나 지시대명사
    elif qcheck_rule_snt(snt_morphs, RULE3_SNT):
        print("rule 3-snt")
        answer = rule_uncertain_obj(
                        atts=detected_atts,
                        dist_centers=detected_dist_centers,
                        att_set=ATTSET_RULE3,
                        )
        
        return answer

    # rule 3-2 : 모양 - 주어가 있는 경우
    elif qcheck_rule_token(snt_morphs, RULE3_TOKEN):
        answer = rule_certain_obj(
                        snt=snt, 
                        objs=detected_objs,
                        atts=detected_atts,
                        att_set=ATTSET_RULE3,
                        rule_tokens=RULE3_TOKEN,
                        tagger=tagger)

        return answer

    # rule 4-1 : 무늬 - 주어가 없거나 지시대명사
    elif qcheck_rule_snt(snt_morphs, RULE4_SNT):
        print("rule 4-snt")
        answer = rule_uncertain_obj(
                        atts=detected_atts,
                        dist_centers=detected_dist_centers,
                        att_set=ATTSET_RULE4,
                        )
        return answer

    # rule 4-2 : 무늬 - 주어가 있는 경우
    elif qcheck_rule_token(snt_morphs, RULE4_TOKEN):
        print("rule 4-token")
        answer = rule_certain_obj(
                    snt=snt, 
                    objs=detected_objs,
                    atts=detected_atts,
                    att_set=ATTSET_RULE4,
                    rule_tokens=RULE4_TOKEN,
                    tagger=tagger)
        return answer
    
    # 개수 
    elif qcheck_rule_token(snt_morphs, RULE5_TOKEN):
        print("rule 5-token")
        answer = rule_certain_obj_count(
                    snt=snt, 
                    objs=detected_objs,
                    tagger=tagger
                    )
        return answer

    # 유무
    elif qcheck_rule_snt(snt_morphs, RULE6_SNT, startswith=False):
        print("rule 6-snt")
        answer = rule_certrain_obj_exist(
                    snt_morphs=snt_morphs,
                    objs=detected_objs,
                    rule_snts=RULE6_SNT,
                    tagger=tagger)
        return answer

    return 'unanswerable'


# Question Type Checkers
def qcheck_rule_snt(snt_morphs:List[str], rule_snts:List[str], startswith=True):
    """
    configs/detction_rule/RULE#_SNT에 등록된 문장 시작/종결 규칙에 따라 판별
    구체적인 대상이 정해지지 않은 경우의 질문 유형에 사용
    e.g)
    rule_snt = [
             '이것 은 무엇',
             '이 건 무엇',
             ...
             ]
    """

    snt_char = ' '.join(snt_morphs)
    
    for r in rule_snts:
        if startswith and snt_char.startswith(r):
            return True
        elif not startswith and snt_char.endswith(r):
            return True
    return False

def qcheck_rule_token(
    snt_morphs:List,
    rule_tokens:Union[str, List[str]]
    ):
    """
    configs/detction_rule/RULE#_TOKEN에 등록된 문장 시작 규칙에 따라 판별
    구체적인 대상이 정해진 경우의 질문 유형에 사용
    e.g) rule_token = ['색','색상']
    """
    def _checker(token_to_check):
        if type(rule_tokens) == str:
            return token_to_check == rule_tokens
        elif type(rule_tokens) == list:
            return any([(token_to_check == rule) for rule in rule_tokens])
        else:
            raise ValueError
        
    for token in snt_morphs:
        if _checker(token):
            return True
    return False

# Rule-Based Answer Functions
def rule_uncertain_obj_what(
               objs: List,
               atts:List,
               dist_centers:List
                ):
    """
    1. (지시대명사) + `무엇` + ㄴ가요?/ㅂ니까?
        - 화면의 중심에 가까운 (att) + obj 출력
        - e.g. 무엇인가요?, 이것은 무엇인가요?, 저건 무엇인가요?, 화면에 무엇이 보이나요?
    """
    
    # 정답 단어 출력
    idx, _ = min(dist_centers, key=lambda x : x[1])
    if atts[idx] != '__no_attribute__':
        return atts[idx] + ' ' + objs[idx] 
    else:
        return objs[idx]
    
def rule_uncertain_obj(
             atts: List,
             dist_centers:List,
             att_set: set,
             ):
    """
    att_set : attributes_vocab.txt에 등록된 `색`(`색깔`), `무늬`, `모양`의 인덱스
    2. 문장에 `색`, `색깔`이 포함된 경우(단, `-색`, `-색깔` 제외)
       *1. 답변할 대상을 특정하지 않는 경우
            - 화면의 중심에 가까운 obj의 att 중에서 색에 대한 것만 출력
            - e.g. 무슨 색이야?
        2. 답변할 대상을 특정한 경우
            - Detection된 obj 목록에서 해당 명사의 att 추출(`endswith`)
            - e.g. 가방 무슨 색이야?
    3. 문장에 `모양`이 포함된 경우
       *1. 답변할 대상을 특정하지 않는 경우
            - 화면의 중심에 가까운 obj의 att 중에서 모양에 대한 것만 출력
            - e.g. 무슨 모양인가요?
        2. 답변할 대상을 특정한 경우
            - Detection된 obj 목록에서 해당 명사의 att 추출(`endswith`)
            - e.g. 바닥은 무슨 모양이야?
    4. 문장에 `무늬`가 포함된 경우
       *1. 답변할 대상을 특정하지 않는 경우
            - 화면의 중심에 가까운 obj의 att 중에서 모양에 대한 것만 출력
            - e.g. 무슨 무늬인가요?
        2. 답변할 대상을 특정한 경우
            - Detection된 obj 목록에서 해당 명사의 att 추출(`endswith`)
            - e.g. 바닥은 무슨 무늬이야?
    """

    # att_set에 있다면 정답 단어 출력
    indices, _ = zip(*sorted(dist_centers, key=lambda x : x[1]))
    for idx in indices:
        if atts[idx] in att_set:
            return atts[idx]
    return 'unanswerable'

def rule_certain_obj(
           snt:str,
           objs: List,
           atts: List,
           att_set: set,
           rule_tokens: Union[str,List[str]],
           tagger):
    """
    att_set : attributes_vocab.txt에 등록된 `색`(`색깔`), `무늬`, `모양`의 인덱스
    2. 문장에 `색`, `색깔`이 포함된 경우(단, `-색`, `-색깔` 제외)
        1. 답변할 대상을 특정하지 않는 경우
            - 화면의 중심에 가까운 obj의 att 중에서 색에 대한 것만 출력
            - e.g. 무슨 색이야?
       *2. 답변할 대상을 특정한 경우
            - Detection된 obj 목록에서 해당 명사의 att 추출(`endswith`)
            - e.g. 가방 무슨 색이야?
    3. 문장에 `모양`이 포함된 경우
        1. 답변할 대상을 특정하지 않는 경우
            - 화면의 중심에 가까운 obj의 att 중에서 모양에 대한 것만 출력
            - e.g. 무슨 모양인가요?
       *2. 답변할 대상을 특정한 경우
            - Detection된 obj 목록에서 해당 명사의 att 추출(`endswith`)
            - e.g. 바닥은 무슨 모양이야?
    4. 문장에 `무늬`가 포함된 경우
        1. 답변할 대상을 특정하지 않는 경우
            - 화면의 중심에 가까운 obj의 att 중에서 모양에 대한 것만 출력
            - e.g. 무슨 무늬인가요?
       *2. 답변할 대상을 특정한 경우
            - Detection된 obj 목록에서 해당 명사의 att 추출(`endswith`)
            - e.g. 바닥은 무슨 무늬이야?
    """
    def _checker(token_to_check):
        if type(rule_tokens) == str:
            return token_to_check == rule_tokens
        elif type(rule_tokens) == list:
            return any([(token_to_check == rule) for rule in rule_tokens])
        else:
            raise ValueError

    snt_pos = tagger.pos(snt)
    tokens, _ = zip(*snt_pos)

    # 명사 탐색 범위 축소
    token_startswith = None
    if (type(rule_tokens) == list):
        token_startswith = rule_tokens[0]
    else:
        token_startswith = rule_tokens
    for idx, token in enumerate(tokens):
        if token.startswith(token_startswith):
            cnt_token_index = idx
            break

    snt_pos_serach_range = snt_pos[max(0, cnt_token_index-3):cnt_token_index+3]
    
    for token, pos in snt_pos_serach_range:
        if (pos == "NNG") or (pos == "NNP") or (pos == "OL"):
            if (token == "이") or _checker(token):
                continue
            # 추출된 명사가 검출된 objs 중에 있고 색상 att이면 색상 출력
            try:
                noun_idx = objs.index(token)
                if atts[noun_idx] in att_set:
                    return atts[noun_idx]
            except:
                continue
                
    return "unanswerable"

def rule_certain_obj_count(
               snt:str,
               objs: List,
               tagger,
               ):
    """
    5. 'N'(은/는/이/가) + `몇` + (수량 단위) + ...?
        -  Detection Model이 찾은 Obj 목록(objs) 중에서 `N`의 개수 카운트
        - e.g. 사람이 몇 명이야?, 개가 몇 마리야?
    """

    snt_pos = tagger.pos(snt)
    tokens, tags = zip(*snt_pos)

    # 명사 탐색 범위 축소
    token_startswith = "몇"
    for idx, token in enumerate(tokens):
        if token.startswith(token_startswith):
            cnt_token_idx = idx
            break
    
    # 단위 명사 추출 및 정제
    cnt_word = tokens[cnt_token_idx + 1]
    if cnt_word not in COUNT_NOUN:
        return 'unanswerable'
    
    if cnt_word.startswith('개') and len(cnt_word) > 1:
        cnt_word = '개'
    elif cnt_word.startswith('매') and len(cnt_word) > 1:
        cnt_word = '매'
    elif cnt_word.startswith('명') and len(cnt_word) > 1:
        cnt_word = '명'

    # '몇'의 앞쪽 토큰에 대해서 개수를 세고자하는 명사 탐색
    snt_pos_serach_range = snt_pos[:cnt_token_idx]
    
    for token, pos in snt_pos_serach_range:
        if (token == '화면') or (token == '앞'):
            continue
        if (pos == "NNG") or (pos == "NNP") or (pos == "OL"):
            if (token == "은") or (token == "이"):
                continue
            if token == '차가':
                token = '차'
            if token == '사람':
                n_obj = sum([
                    (obj == '남성') or 
                    (obj == '남자') or 
                    (obj == '여성') or 
                    (obj == '여자') or 
                    (obj.endswith('사람'))
                    for obj in objs])
            elif (token == '남자') or (token == '남성'):
                n_obj = sum([
                    (obj == '남성') or 
                    (obj == '남자')
                    for obj in objs])
            elif (token == '여자') or (token == '여성'):
                n_obj = sum([
                    (obj == '여성') or 
                    (obj == '여자')
                    for obj in objs])
            else:
                n_obj = sum([obj.endswith(token) for obj in objs])

            return ' '.join([str(n_obj), cnt_word])
                
    return "unanswerable"

def rule_certrain_obj_exist(
                            snt_morphs:List[str],
                            objs: List,
                            rule_snts:List[str],
                            tagger,
                            ):

    snt_char = ' '.join(snt_morphs)

    for r in rule_snts:
        yn_obj_idx = snt_char.find(r)
        if yn_obj_idx != -1:
            break
    if yn_obj_idx == -1:
        return "unanswerable"
    
    snt_pos = tagger.pos(snt_char[:yn_obj_idx])

    for token, pos in snt_pos:
        if (token == '화면') or (token == '앞'):
            continue
        if (pos == "NNG") or (pos == "NNP") or (pos == "OL"):
            if token == '사람':
                n_obj = sum([
                    (obj == '남성') or 
                    (obj == '남자') or 
                    (obj == '여성') or 
                    (obj == '여자') or 
                    (obj.endswith('사람'))
                    for obj in objs])
            elif (token == '남자') or (token == '남성'):
                n_obj = sum([
                    (obj == '남성') or 
                    (obj == '남자')
                    for obj in objs])
            elif (token == '여자') or (token == '여성'):
                n_obj = sum([
                    (obj == '여성') or 
                    (obj == '여자')
                    for obj in objs])
            else:
                # if token not in objs:
                #     continue
                n_obj = sum([obj == token for obj in objs])

            if n_obj == 0:
                return '없어'
            else:
                return '있어'

    return "unanswerable"

if __name__ == "__main__":
    from konlpy.tag import Kkma
    from configs.detection_rule import (
        RULE1_SNT,
        RULE2_SNT,
        RULE3_SNT,
        RULE4_SNT,
        RULE2_TOKEN,
        RULE3_TOKEN,
        RULE4_TOKEN,
        RULE5_TOKEN
    )
    kkma = Kkma() 
    q = kkma.morphs('뭐야?')
    print(qcheck_rule_snt(q, RULE1_SNT))
    print(qcheck_rule_snt(q, RULE2_SNT))
    print(qcheck_rule_snt(q, RULE3_SNT))
    print(qcheck_rule_snt(q, RULE4_SNT))
    print(qcheck_rule_token(q, RULE2_TOKEN))
    print(qcheck_rule_token(q, RULE3_TOKEN))
    print(qcheck_rule_token(q, RULE4_TOKEN))