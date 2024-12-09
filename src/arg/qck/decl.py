from abc import ABC, abstractmethod
from typing import NamedTuple, List, Tuple, Union

from data_generator.tokenizer_wo_tf import tokenize_from_tokens


class QCKQuery(NamedTuple):
    query_id: str
    text: str
    def get_id(self):
        return self.query_id


class QCKCandidate(NamedTuple):
    id: str
    text: str

    def get_tokens(self, tokenizer):
        return tokenizer.tokenize(self.text)

    def get_id(self):
        return self.id

    def light_rep(self):
        return QCKCandidate(self.id, "")


class QCKQueryWToken(NamedTuple):
    query_id: str
    text: str
    tokens: List[str]

    def get_tokens(self, tokenizer):
        return self.tokens


class QCKCandidateWToken(NamedTuple):
    id: str
    text: str
    tokens: List[str]

    def get_tokens(self, tokenizer):
        return self.tokens

    def light_rep(self):
        return QCKCandidate(self.id, "")

    @classmethod
    def from_qck_candidate(cls, tokenizer, c: QCKCandidate):
        return QCKCandidateWToken(c.id, c.text, tokenizer.tokenize(c.text))


class KnowledgeDocument(NamedTuple):
    doc_id: str
    tokens: List[str]


class KnowledgeDocumentPart(NamedTuple):
    doc_id: str
    passage_idx: int
    start_location: int
    tokens: List[str]

    def getstate(self):
        return self.doc_id, self.passage_idx, self.start_location, self.tokens

    @classmethod
    def from_state(cls, state):
        return KnowledgeDocumentPart(*state)

    def to_str(self) -> str:
        return "{}_{}".format(self.doc_id, self.passage_idx)


class KDPWToken(NamedTuple):
    doc_id: str
    passage_idx: int
    start_location: int
    tokens: List[str]
    sub_tokens: List[str]

    def get_tokens(self):
        return self.sub_tokens


KD = KnowledgeDocument
KDP = KnowledgeDocumentPart

# KDP , BERT tokenized


class KDP_BT(NamedTuple):
    doc_id: str
    passage_idx: int
    start_location: int
    tokens: List[str]

    def getstate(self):
        return self.doc_id, self.passage_idx, self.start_location, self.tokens

    @classmethod
    def from_state(cls, state):
        return KnowledgeDocumentPart(*state)


class QKInstance(NamedTuple):
    query_text: str
    doc_tokens: List[str]
    data_id: int
    is_correct: int


class QKRegressionInstance(NamedTuple):
    query_text: str
    doc_tokens: List[str]
    data_id: int
    score: float


class QCInstance(NamedTuple):
    query_text: str
    candidate_text: str
    data_id: int
    is_correct: int


class QCInstanceTokenized(NamedTuple):
    query_text: List[str]
    candidate_text: List[str]
    data_id: int
    is_correct: int


class QCKInstance(NamedTuple):
    query_text: str
    candidate_text: str
    doc_tokens: List[str]
    data_id: int
    is_correct: int


class QCKInstanceTokenized(NamedTuple):
    query_text: List[str]
    candidate_text: List[str]
    doc_tokens: List[str]
    is_correct: int
    data_id: int


class CKInstance(NamedTuple):
    candidate_text: str
    doc_tokens: List[str]
    data_id: int
    is_correct: int


QKUnit = Tuple[QCKQuery, List[KDP]]

QKUnitWToken = Tuple[QCKQueryWToken, List[KDPWToken]]

QKUnitBT = Tuple[QCKQuery, List[KDP_BT]]


def add_tokens_to_qk_unit(qk_unit: QKUnit, tokenizer) -> QKUnitWToken:
    query, kdp_list = qk_unit
    q = QCKQueryWToken(query.query_id, query.text, tokenizer.tokenize(query.text))
    new_kdp_list = []
    for kdp in kdp_list:
        sub_tokens = tokenize_from_tokens(tokenizer, kdp.tokens)
        kdp_w_tokens = KDPWToken(kdp.doc_id, kdp.passage_idx, kdp.start_location, kdp.tokens, sub_tokens)
        new_kdp_list.append(kdp_w_tokens)
    return q, new_kdp_list


class PayloadAsTokens(NamedTuple):
    passage: List[str]
    text1: List[str]
    text2: List[str]
    data_id: int
    is_correct: int


class PayloadAsIds(NamedTuple):
    passage: List[int]
    text1: List[int]
    text2: List[int]
    data_id: int
    is_correct: int


def get_qk_pair_id(entry) -> Tuple[str, str]:
    return entry['query'].query_id, "{}_{}".format(entry['kdp'].doc_id, entry['kdp'].passage_idx)


def get_qc_pair_id(entry) -> Tuple[str, str]:
    return entry['query'].query_id, entry['candidate'].id


class FormatHandler(ABC):
    @abstractmethod
    def get_pair_id(self, entry):
        pass

    @abstractmethod
    def get_mapping(self):
        pass

    @abstractmethod
    def drop_kdp(self):
        pass


class QCKFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qc_pair_id(entry)

    def get_mapping(self):
        return qck_convert_map

    def drop_kdp(self):
        return True


class QCFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qc_pair_id(entry)

    def get_mapping(self):
        return qc_convert_map

    def drop_kdp(self):
        return False


class QKFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qk_pair_id(entry)

    def get_mapping(self):
        return qk_convert_map

    def drop_kdp(self):
        return False


class QCKLFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qc_pair_id(entry)

    def get_mapping(self):
        return qckl_convert_map

    def drop_kdp(self):
        return False


class QCWTFormatHandler(FormatHandler):
    def get_pair_id(self, entry):
        return get_qc_pair_id(entry)

    def get_mapping(self):
        return qcwt_convert_map

    def drop_kdp(self):
        return False


def get_format_handler(input_type):
    if input_type == "qck":
        return QCKFormatHandler()
    elif input_type == "qc":
        return QCFormatHandler()
    elif input_type == "qk":
        return QKFormatHandler()
    elif input_type == "qckl":
        return QCKLFormatHandler()
    elif input_type == "qcwt":
        return QCWTFormatHandler()
    else:
        assert False


qck_convert_map = {
        'kdp': KDP,
        'query': QCKQuery,
        'candidate': QCKCandidate
    }
qk_convert_map = {
        'kdp': KDP,
        'query': QCKQuery,
    }

qc_convert_map = {
        'query': QCKQuery,
        'candidate': QCKCandidate,
    }

qcwt_convert_map = {
        'query': QCKQuery,
        'candidate': QCKCandidateWToken,
    }


def parse_kdp_list(*tuple):
    l = list(tuple)
    return list([KDP(*kdp) for kdp in l])


QCKQueryLike = Union[QCKQuery, QCKQueryWToken]
KDPLike = Union[KDP, KDPWToken]
QCKCandidateLike = Union[QCKCandidate, QCKCandidateWToken]


def get_light_qckquery(query: QCKQueryLike):
    return QCKQuery(query.query_id, "")


def get_light_qckcandidate(c: QCKCandidateLike):
    return QCKCandidate(c.id, "")


def get_light_kdp(k: KDPLike):
    return KnowledgeDocumentPart(k.doc_id, k.passage_idx, k.start_location, [])





qckl_convert_map = {
        'kdp_list': parse_kdp_list,
        'query': QCKQuery,
        'candidate': QCKCandidate
    }

class QCKOutEntry(NamedTuple):
    logits: List[float]
    query: QCKQuery
    candidate: QCKCandidate
    kdp: KDP

    @classmethod
    def from_dict(cls, d):
        return QCKOutEntry(d['logits'], d['query'], d['candidate'], d['kdp'])