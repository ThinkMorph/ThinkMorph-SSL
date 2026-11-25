from typing import Dict, List

import jsonlines


class DataUtilities:
    @staticmethod
    def load_jsonlines(path: str) -> List[Dict[str, List]]:
        """Read a local JSON Lines file into a list of rows."""
        data_list = []
        with jsonlines.open(path, "r") as f:
            for data in f:
                data_list.append(data)
        return data_list
