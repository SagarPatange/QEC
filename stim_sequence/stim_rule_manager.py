
# =====================================================================
# stim_rule_manager.py
# ---------------------------------------------------------------------
# Ordered rule registry compatible with existing Rule/RuleManager ergonomics.
# =====================================================================

from __future__ import annotations
from typing import List

from stim_memory_manager import StimMemoryManager, StimMemoryInfo


class StimRuleManager:
    def __init__(self, memory_manager: StimMemoryManager):
        self._mm = memory_manager
        self._rules: List = []

    def __len__(self) -> int:
        return len(self._rules)

    def __getitem__(self, i: int):
        return self._rules[i]

    def get_memory_manager(self) -> StimMemoryManager:
        return self._mm

    def load(self, rule) -> None:
        if not hasattr(rule, "priority"):
            raise AttributeError("Rule must have a 'priority' attribute (int).")
        idx = 0
        while idx < len(self._rules) and self._rules[idx].priority <= rule.priority:
            idx += 1
        self._rules.insert(idx, rule)

    def expire(self, rule) -> None:
        try:
            self._rules.remove(rule)
        except ValueError:
            pass

    def evaluate(self, resource_manager) -> None:
        for rule in list(self._rules):
            if hasattr(rule, "select") and callable(rule.select):
                selected = list(rule.select(resource_manager, self._mm))
            else:
                if not hasattr(rule, "is_valid") or not callable(rule.is_valid):
                    continue
                selected = [info for info in self._mm if rule.is_valid(info)]
            if not selected:
                continue
            if hasattr(rule, "do") and callable(rule.do):
                rule.do(resource_manager, selected)
