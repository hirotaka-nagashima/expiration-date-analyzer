import math
import os
import re
from collections.abc import Iterable

import ipadic
import MeCab
import numpy as np
from sklearn import model_selection, pipeline
from sklearn.feature_extraction import text

from analyzer.labeler import labeler
from utils import jsonhandler, timenormalizer

X = tuple[str, timenormalizer.TimeExpressions]
Y = labeler.Label
_X = text.CountVectorizer
_Y = bool

Result = tuple[Iterable[Y], Iterable[Y]]


class Wrapper:
    """Wrapper of scikit-learn."""

    def __init__(self, classifier, text_divider):
        self._clf = pipeline.Pipeline(
            [
                ("vect", text.CountVectorizer(analyzer=text_divider.extract_words)),
                ("clf", classifier),
            ]
        )

    def cross_validate(
        self, x: list[X], y: list[Y], y_true: list[Y] | None = None, size_train=None
    ):
        """
        Args:
            x: X.
            y: Y. If y_true is given, used only for training.
            y_true: Y used only for testing.
            size_train: You can specify a size of training data. None uses
                an 80% of all data as training data.
        """
        if y_true is None:
            y_true = y

        def to_bool(a):
            return [False if o is None else True for o in a]

        y = to_bool(y)
        x, y, y_true = np.array(x), np.array(y), np.array(y_true)

        # Cross-validate manually because of distinction between y to train and
        # y to test: y_train is obtained from AutoLabeler but y_test is obtained
        # from HandLabeler.
        results: list[Result] = []
        kf = model_selection.KFold(shuffle=True, random_state=0)
        kf.get_n_splits(x)
        for train_index, test_index in kf.split(x):
            if size_train is not None:
                test_index = np.concatenate([test_index, train_index[size_train:]])
                train_index = train_index[:size_train]
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y_true[test_index]
            self._clf.fit(x_train, y_train)
            y_pred = self._clf.predict(x_test)
            y_pred = np.array(Wrapper._restore_datetime(x_test, y_pred))
            results.append((y_test, y_pred))

        Wrapper._show_result(results)

    @staticmethod
    def _restore_datetime(x, y_bool: list[_Y]) -> list[Y]:
        y_time = []
        for (_, time_expressions), y_bool_ in zip(x, y_bool):
            y_time_ = None
            if y_bool_:
                # Restore datetime.
                for _, (since, until) in time_expressions:
                    if until is None:  # definite time
                        y_time_ = since if y_time_ is None else max(since, y_time_)
                if y_time_ is None:  # only dates
                    for _, (_, until) in time_expressions:
                        y_time_ = until if y_time_ is None else max(until, y_time_)
            y_time.append(y_time_)
        return y_time

    @staticmethod
    def _show_result(results: list[Result]):
        # Calculate a confusion matrix.
        ttp, tp, fp, fn, tn = 0, 0, 0, 0, 0
        for result in results:
            for y_test, y_pred in zip(result[0], result[1]):
                if y_test is None and y_pred is None:
                    tn += 1
                if y_test is None and y_pred is not None:
                    fp += 1
                if y_test is not None and y_pred is None:
                    fn += 1
                if y_test is not None and y_pred is not None:
                    tp += 1
                    if y_test == y_pred:  # as datetime
                        ttp += 1

        try:
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        except ZeroDivisionError:
            mcc = np.NaN

        print("TTP", "TP", "FP", "FN", "TN", sep="\t")
        print(ttp, tp, fp, fn, tn, sep="\t")
        print(f"MCC: {mcc}")


class TextDivider:
    """Divides a text into meaningful words."""

    _PATH = os.path.abspath(__file__ + "/../../../data/words.json")
    _CHASEN_ARGS = (
        r' -F "%m\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"'
        r' -U "%m\t%m\t%m\t%F-[0,1,2,3]\t\t\n"'
    )
    _tagger = MeCab.Tagger(ipadic.MECAB_ARGS + _CHASEN_ARGS)

    def __init__(
        self,
        parsed_texts_for_tf_table: list[X] | None = None,
        lower_bound=0,
        pos_include=None,
        top_k=None,
        top_k_type="frequentWords",
    ):
        """
        Args:
            parsed_texts_for_tf_table: List of
                tuple[str, timenormalizer.TimeExpressions] used to construct a
                TF table. The table filters words according to lower_bound.
            lower_bound: Only words whose TF values are greater or equal to this
                value will be extracted.
            pos_include: Given this, filters words by POS. All POS are
                "その他", "フィラー", "感動詞", "記号", "形容詞", "助詞", "助動詞",
                "接続詞", "接頭詞", "動詞", "副詞", "名詞", "連体詞".
            top_k: Given this, with a top_k_type method, extracts only top_k
                words.
            top_k_type: "frequentWords" or "importantWordsPred" or
                "importantWordsTrue".
        """
        self._tf: dict[str, int] = {}
        self._tf_is_locked = True
        self._lower_bound = lower_bound
        self._pos_include = pos_include
        self._words_include = None

        # Construct a TF table.
        if parsed_texts_for_tf_table is not None:
            for parsed_text in parsed_texts_for_tf_table:
                words = self.extract_words(parsed_text)
                for word in words:
                    if word not in self._tf:
                        self._tf[word] = 0
                    self._tf[word] += 1
            self._tf_is_locked = False

        # Load words ordered by importance.
        if top_k is not None:
            words = jsonhandler.load(TextDivider._PATH)[top_k_type]
            self._words_include = {w for _, w in words[:top_k]}

    def extract_words(self, parsed_text: X):
        # First, replace URLs and time expressions with a dummy.
        dummy = "DUMMY"
        text_, time_expressions = parsed_text
        text_ = re.sub(r"https://t\.co/\w+", dummy, text_)
        for keyword, _ in time_expressions:
            text_ = text_.replace(keyword, dummy)

        info = getattr(TextDivider._tagger, "parse")(text_)
        info = [line.split("\t") for line in info.split("\n")]

        # Filter words.
        base_forms = []
        for line in info:
            if len(line) < 4:
                continue
            base_form = line[2]
            pos = line[3].split("-")[0]

            # Exclude a dummy.
            if dummy in base_form:
                continue
            # Exclude a rare word.
            if not self._tf_is_locked:
                if self._tf.get(base_form, 0) < self._lower_bound:
                    continue
            # Exclude by POS.
            if self._pos_include is not None:
                if pos not in self._pos_include:
                    continue
            # Exclude by exceptions.
            if self._words_include is not None:
                if base_form not in self._words_include:
                    continue

            base_forms.append(base_form)

        return base_forms
