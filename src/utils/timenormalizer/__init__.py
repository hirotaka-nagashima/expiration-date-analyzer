import datetime as dt
from typing import List, Optional, Tuple

# NOTE: TimeExpressions should not be defined as Dict[str, Duration] because
# same words may express different time, which is specification of goo service.
Duration = Tuple[dt.datetime, Optional[dt.datetime]]
TimeExpression = Tuple[str, Duration]
TimeExpressions = List[TimeExpression]

__author__ = "Nagashima Hirotaka"
__all__ = ["en", "ja", "Duration", "TimeExpression", "TimeExpressions"]
