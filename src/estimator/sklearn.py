import re
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import MeCab
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn.feature_extraction import text

from analyzer.labeler import labeler
from utils import jsonhandler
from utils import timenormalizer

X = Tuple[str, timenormalizer.TimeExpressions]
Y = labeler.Label
_X = text.CountVectorizer
_Y = bool

_Result = Tuple[Iterable[_Y], Iterable[_Y]]


class Wrapper:
    """Wrapper of scikit-learn."""

    def __init__(self, classifier, text_divider):
        self._clf = pipeline.Pipeline([
            ("vect", text.CountVectorizer(analyzer=text_divider.extract_words)),
            ("clf", classifier),
        ])

    def cross_validate(self, x: List[X], y: List[Y],
                       y_true: Optional[List[Y]] = None,
                       shows_only_counts=False) -> float:
        """
        Args:
            x: X.
            y: Y. If y_true is given, used only for training.
            y_true: Y used only for testing.
            shows_only_counts: Shows only counts in a confusion matrix. The
                order is TP, FP, FN, TN.

        Returns:
            Mean Matthews correlation coefficient.
        """
        if y_true is None:
            y_true = y

        def to_bool(a): return [False if o is None else True for o in a]
        y, y_true = to_bool(y), to_bool(y_true)
        x, y, y_true = np.array(x), np.array(y), np.array(y_true)

        # Cross-validate manually because of distinction between y to train and
        # y to test: y_train is obtained from AutoLabeler but y_test is obtained
        # from HandLabeler.
        results = []  # type: List[_Result]
        kf = model_selection.KFold(shuffle=True, random_state=0)
        kf.get_n_splits(x)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y_true[test_index]
            self._clf.fit(x_train, y_train)
            y_pred = self._clf.predict(x_test)
            results.append((y_test, y_pred))

        final_result = Wrapper._concat_results(results)
        if shows_only_counts:
            Wrapper._show_counts(final_result)
        else:
            Wrapper._show_results(results, final_result=final_result)

        mcc = metrics.matthews_corrcoef(*final_result)
        return mcc

    def evaluate(self, x_train: List[X], x_test: List[X],
                 y_train: List[Y], y_test: List[Y]):
        def to_bool(a): return [False if o is None else True for o in a]
        y_train, y_test = to_bool(y_train), to_bool(y_test)
        self._clf.fit(x_train, y_train)
        y_pred = self._clf.predict(x_test)
        result = (y_test, y_pred)
        Wrapper._show_result(result)

    @staticmethod
    def _concat_results(results: List[_Result]) -> _Result:
        all_y_true, all_y_pred = [], []
        for y_true, y_pred in results:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
        return all_y_true, all_y_pred

    @staticmethod
    def _show_result(result: _Result):
        print("TP", "FP", "FN", "TN", sep="\t")
        Wrapper._show_counts(result)
        print(f"MCC: {metrics.matthews_corrcoef(*result)}")

    @staticmethod
    def _show_results(results: List[_Result],
                      final_result: Optional[_Result] = None):
        if final_result is None:
            final_result = Wrapper._concat_results(results)

        print("TP", "FP", "FN", "TN", sep="\t")
        Wrapper._show_counts(final_result)
        for result in results:
            Wrapper._show_counts(result)
        print(f"MCC: {metrics.matthews_corrcoef(*final_result)}")

    @staticmethod
    def _show_counts(result: _Result):
        tn, fp, fn, tp = metrics.confusion_matrix(*result).ravel()
        print(tp, fp, fn, tn, sep="\t")


class TextDivider:
    """Divides a text into meaningful words."""

    _PATH = r"C:\Users\Admin\Cloud\Programs\IntelliJ IDEA\Study\Best By Dates Analyzer\data\words.json"
    _tagger = MeCab.Tagger("-Ochasen")

    def __init__(self, parsed_texts_for_tf_table: Optional[List[X]] = None,
                 lower_bound=0, pos_include=None, top_k=None,
                 top_k_type="frequentWords"):
        """
        Args:
            parsed_texts_for_tf_table: List of
                Tuple[str, timenormalizer.TimeExpressions] used to construct a
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
        self._tf = {}  # type: Dict[str, int]
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
