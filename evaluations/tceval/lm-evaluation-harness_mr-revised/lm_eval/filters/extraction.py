import re

from lm_eval.api.filter import Filter


class RegexFilter(Filter):
    """ """

    def __init__(
        self, regex_pattern: str = r"#### (\-?[0-9\.\,]+)", fallback: str = "[invalid]"
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.fallback = fallback

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.search(resp)
                if match:
                    match = match.group(1).strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        # print(resps)
        filtered_resps = list(map(lambda x: filter_set(x), resps))
        # print(filtered_resps)

        return filtered_resps


class RegexFilterWFindall(RegexFilter):
    """ """

    def __init__(
        self, 
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)", fallback: str = "[invalid]", 
        take_which: str = "first"
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        super().__init__(regex_pattern, fallback)

        assert take_which in {"first", "last"}
        self._take_i = {"first": 0, "last": -1}[take_which]

    def apply_find_all(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                cands = self.regex.findall(resp)
                match = self.fallback
                # Take the first found

                cand_matches = [opt.strip() for cand in cands for opt in cand if len(opt)]
                if len(cand_matches):
                    match = cand_matches[self._take_i]

                filtered.append(match)
            return filtered

        # print(resps)
        filtered_resps = list(map(lambda x: filter_set(x), resps))
        # print(filtered_resps)

        return filtered_resps

    def apply(self, resps, docs):
        return self.apply_find_all(resps, docs)

class WhitespaceFilter(Filter):
    """ """

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                if resp.startswith(" "):
                    resp = resp[1:]

                filtered_resp.append(resp)

            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps
