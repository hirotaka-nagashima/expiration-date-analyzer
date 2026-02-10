import datetime as dt

# NOTE: TimeExpressions should not be defined as dict[str, Duration] because
# same words may express different time, which is specification of goo service.
Duration = tuple[dt.datetime, dt.datetime | None]
TimeExpression = tuple[str, Duration]
TimeExpressions = list[TimeExpression]

__author__ = "Nagashima Hirotaka"
__all__ = ["en", "ja", "Duration", "TimeExpression", "TimeExpressions"]
