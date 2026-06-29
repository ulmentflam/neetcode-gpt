from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens: List[str] = list(corpus)
        merges: List[List[str]] = []
        for _ in range(num_merges):
            n: int = len(tokens)
            if n < 2: # We must have at least a pair of tokens to merge
                break
            
            # Count adj pairs
            pairs: dict = {}
            most_frequent = 0
            for i in range(n - 1):
                pair = (tokens[i], tokens[i + 1])
                count = pairs[pair] = pairs.get(pair, 0) + 1
                if count > most_frequent:
                    most_frequent = count 
            
            if not pairs:
                break
            
            # Find the most frequent pair lexicograpically smallest first.
            candidates = sorted(p for p,c in pairs.items() if c == most_frequent)
            best = candidates[0]

            merges.append([best[0], best[1]])

            # Merge all non-overlapping occurences from left to right. 
            new_tokens: List[str] = []
            i = 0
            while i < n:
                if i < n - 1 and tokens[i] == best[0] and tokens[i+1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                    continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens

        return merges

