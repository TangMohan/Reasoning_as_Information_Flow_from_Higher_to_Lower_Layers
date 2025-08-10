from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import itertools
import re
import math
import torch
from pysat.solvers import Glucose4
from pysat.formula import CNF
import heapq

class AddDataset(Dataset):
    def __init__(self, length, max_len):
        self.length = length
        self.max_len = max_len
        self.data = []
        self.labels = []

        for _ in range(length):
            len1 = random.randint(1, max_len)
            len2 = random.randint(1, max_len)
            num1 = ''.join(str(random.randint(0, 9)) for _ in range(len1))
            num2 = ''.join(str(random.randint(0, 9)) for _ in range(len2))
            sum = ' '.join(str(int(num1) + int(num2)))
            num1 = ' '.join(num1)
            num2 = ' '.join(num2)
            self.data.append(f"{num1[::-1]} + {num2[::-1]}")
            self.labels.append(sum[::-1])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'instruction' : "Compute addition.", "input":self.data[idx], "output": str(self.labels[idx])}

class ParityDatasetPositionLabeled(Dataset):
    def __init__(self, length, seq_len):
        self.length = length
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        for _ in range(length):
            len = random.randint(1, seq_len)
            binary_str = ''.join(str(torch.randint(2, size=()).item()) for _ in range(len)) #len
            #for _ in range(seq_len - len):
            #  binary_str = binary_str + "0"
            data = ""
            for i in range(len):
            #for i in range(seq_len):
              data = data + str(i) + ": " + binary_str[i] + " "
            self.data.append(data)
            self.labels.append(sum(int(bit) for bit in binary_str) % 2)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'instruction' : "Output the parity of this sequence.", "input":self.data[idx], "output": str(self.labels[idx])}

class MultistepArithmeticTask:
    """A simple text-based arithmetic task of variable difficulty."""

    def __init__(
        self,
        num_trials=100,
        verbose=False,
        operations=["+", "-"],
        numbers=list(range(-9,10)),#list(range(-9, 10)),
        with_spaces=True,
        depth_level_list=([], ),#([], [2], [2, 2], [2, 2, 2]),
        lengths=list(range(2, 4))#list(range(2, 5)),
    ):
        """Defines a simple arithmetic task of varying difficulty levels.

        Args:
            seed: Numpy random seed.
            num_trials: Number of examples to evaluate with each configuration of depth_levels and length.
            verbose: Print each example during evaluation.
            operations: List of operations from which one samples randomly, for example ['+', '-', '*'].
            numbers: List of numbers from which one samples randomly to appear in the question string.
            with_spaces: Whether the input string contains spaces between numbers and operations.
            depth_level_list: How many parenthesis are in each level of the tree-like structure.
                For example [2,3] means the string contains two main parenthesis, each with three subparenthesis.
            lengths: number of numbers to be operated on in the inner-most parenthesis level.
        Questions are of the form: '(2 * 4 - 2) = ' and '((1 + 1) + (3 + 2)) ='.
        Answers: '4' and '7'  etc.
        """
        self.num_trials = num_trials
        self.verbose = verbose
        self.operations = operations
        self.numbers = [str(x) for x in numbers]
        self.with_spaces = with_spaces
        self.depth_level_list = depth_level_list
        self.lengths = lengths

    def generate_content(self, length, rng = None):
        substr = "("
        space = " " if self.with_spaces else ""
        for _ in range(length - 1):
            substr = (
                substr
                + random.choice(self.numbers)
                + space
                + random.choice(self.operations)
                + space
            )
        substr = substr + random.choice(self.numbers) + ")"
        return substr

    def generate_subparenthesis(self, subparenthesis, length):
        if not subparenthesis:
            return self.generate_content(length)
        else:
            space = " " if self.with_spaces else ""
            substr = "("
            for _ in range(subparenthesis[0] - 1):
                substr = (
                    substr
                    + self.generate_subparenthesis(subparenthesis[1:], length)
                    + space
                    + random.choice(self.operations)
                    + space
                )
            substr = (
                substr + self.generate_subparenthesis(subparenthesis[1:], length) + ")"
            )
            return substr

    def generate_string(self, depth_levels, length):
        return self.generate_subparenthesis(depth_levels, length)

class Multiple_Arithmetic_Dataset(Dataset):
    def __init__(self, trials):
        task = MultistepArithmeticTask()
        self.data = []
        self.labels = []
        for _ in range(trials):
            depth_levels = random.choice(task.depth_level_list)
            length = random.choice(task.lengths)
            input = task.generate_string(depth_levels=depth_levels, length=length)
            problem = input + " = "
            self.data.append(problem)
            self.labels.append(" ".join(str(eval(input) % 10)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'instruction' : "Evaluate the expression in modulo 10.", "input":self.data[idx], "output": str(self.labels[idx])}

def generate_instructions(max_steps_per_instruction, max_total_instructions):
    instructions = []
    num = 0
    while num < max_total_instructions:
        steps = random.randint(1, max_steps_per_instruction)
        direction = random.choices(
            ["Take {} steps".format(steps), "Turn right", "Turn left", "Turn around"],
            weights=[0.52, 0.16, 0.16, 0.16],
            k=1
        )[0]
        if direction.startswith("Take"):
          num = num + 1
        instructions.append(direction)
    return instructions

def follow_instructions(instructions):
    x, y = 0, 0
    direction = 0  # 0: up, 1: right, 2: down, 3: left
    for instr in instructions:
        if instr.startswith("Take"):
            steps = int(instr.split()[1])
            if direction == 0:
                y += steps
            elif direction == 1:
                x += steps
            elif direction == 2:
                y -= steps
            elif direction == 3:
                x -= steps
        elif instr == "Turn right":
            direction = (direction + 1) % 4
        elif instr == "Turn left":
            direction = (direction - 1) % 4
        elif instr == "Turn around":
            direction = (direction + 2) % 4
    # print(f"{x}, {y}")
    return x % 2 == 0 and y % 2 == 0


class NavigateDataset(Dataset):
    def __init__(self, length, max_steps_per_instruction, max_total_instructions):
        self.length = length
        self.data = []
        self.labels = []
        target_per_label = math.ceil(length/(max_total_instructions - 1)/2)
        # print(target_per_label)
        data = []
        for num_total_instructions in range(2, max_total_instructions + 1):
          data_true = []
          data_false = []

          while len(data_true) < target_per_label or len(data_false) < target_per_label:
            # print(f"True: {len(data_true)}")
            # print(f"False: {len(data_false)}")
            instructions = generate_instructions(max_steps_per_instruction, num_total_instructions)
            question = " ".join(instructions)
            answer = follow_instructions(instructions)

            if answer and len(data_true) < target_per_label:
                data_true.append({"Q": question, "A": answer})
            elif not answer and len(data_false) < target_per_label:
                data_false.append({"Q": question, "A": answer})

          data.extend( data_true + data_false)
        random.shuffle(data)
        data = data[:length]
        for sample in data:
            self.data.append(sample["Q"])
            self.labels.append(str(sample["A"]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'instruction' : "If you follow these instructions, do you return to the starting point?", "input":self.data[idx], "output": str(self.labels[idx])}

class Multiplication_Dataset(Dataset):
    def __init__(self, trials):
        task = MultistepArithmeticTask(operations = ["*"], numbers=list(range(0,11)))
        self.data = []
        self.labels = []
        for depth_levels, length in itertools.product(
            task.depth_level_list, task.lengths
        ):
          for _ in range(trials):
            input = task.generate_string(depth_levels=depth_levels, length=length)
            problem = input + " = "
            self.data.append(problem)
            self.labels.append(" ".join(str(eval(input) % 11)))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'instruction' : "Evaluate the expression (python version) in modulo 11.", "input":self.data[idx], "output": str(self.labels[idx])}


class RandomPruferExpressionDataset(Dataset):
    """
    A dataset that:
      - For each example, chooses a random n in [min_n, max_n].
      - Generates a random labeled tree on {0,1,...,n-1} via a random Prüfer code (0-based).
      - Assigns each node:
          * A numeric label (from `numbers`)
          * A random sign in {+, -}
          * A random operator in {+, *} (used if the node has >=1 child)
      - Interprets the tree as rooted at node 0,
        then builds an expression string from that root.
      - Evaluates the expression (mod 10).
    """

    def __init__(
        self,
        size=1000,         # how many expressions to generate
        min_n=1,           # minimum number of nodes
        max_n=6,           # maximum number of nodes
        numbers=range(1, 10)  # possible numeric labels
    ):
        """
        Args:
            size (int): number of expressions (trees) to generate
            min_n (int): minimum number of nodes
            max_n (int): maximum number of nodes
            numbers (iterable): possible numeric labels for each node
        """
        super().__init__()
        self.size = size
        self.min_n = min_n
        self.max_n = max_n
        self.numbers = list(numbers)

        self.data = []
        self.labels = []

        for _ in range(self.size):
            # 1) Sample a random n
            n = random.randint(self.min_n, self.max_n)

            # Edge case: if n == 1, the tree is just a single node
            #   * There's no Prüfer code for n=1, but let's handle it.
            if n == 1:
                # Just a single node: expression is sign + value
                node_info = {
                    0: {
                        "value": random.choice(self.numbers),
                        "sign": random.choice(["+", "-"]),
                        "operator": random.choice(["+", "*"])
                    }
                }
                # Expression for a single node
                expr_str = self._leaf_expr(node_info[0])
            else:
                # 2) Generate a random Prüfer code (0-based) of length n-2
                prufer = [random.randint(0, n-1) for _ in range(n-2)]
                # 3) Decode into edges
                edges = self._decode_prufer(prufer, n)
                # 4) Build adjacency
                adjacency = self._edges_to_adjacency(edges, n)

                # 5) Assign random numeric label, sign, operator to each node
                node_info = {}
                for node_id in range(n):
                    node_info[node_id] = {
                        "value": random.choice(self.numbers),
                        "sign": random.choice(["+", "-"]),
                        "operator": random.choice(["+", "*"])
                    }

                # 6) Root the tree at node 0 and build expression
                expr_str = self._build_expression_string(
                    current=0,
                    parent=None,
                    adjacency=adjacency,
                    node_info=node_info
                )

            # 7) Evaluate expression in Python, mod 10
            try:
                val = eval(expr_str)
                val_mod_10 = val % 10
            except Exception:
                val_mod_10 = 0

            # 8) Store into dataset
            self.data.append(expr_str + " = ")
            self.labels.append(str(val_mod_10))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "instruction": "Evaluate this expression modulo 10.",
            "input": self.data[idx],
            "output": self.labels[idx],
        }

    # ----------------------------------------------------------------
    # (A) Decode Prüfer code -> edges (0-based)
    # ----------------------------------------------------------------
    def _decode_prufer(self, prufer, n):
        """
        Decode a 0-based Prüfer code of length (n-2) 
        into a list of edges on {0,1,...,n-1}.
        """
        freq = [0] * n
        for x in prufer:
            freq[x] += 1

        # min-heap of leaves
        leaf_queue = []
        for v in range(n):
            if freq[v] == 0:
                heapq.heappush(leaf_queue, v)

        edges = []
        for x in prufer:
            u = heapq.heappop(leaf_queue)
            edges.append((u, x))
            freq[x] -= 1
            if freq[x] == 0:
                heapq.heappush(leaf_queue, x)

        # last two vertices
        u = heapq.heappop(leaf_queue)
        w = heapq.heappop(leaf_queue)
        edges.append((u, w))

        return edges

    def _edges_to_adjacency(self, edges, n):
        """
        Build adjacency from edges. adjacency[u] = [neighbors...].
        """
        adjacency = {i: [] for i in range(n)}
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        return adjacency

    # ----------------------------------------------------------------
    # (B) Build Expression from Root
    # ----------------------------------------------------------------
    def _build_expression_string(self, current, parent, adjacency, node_info):
        """
        DFS-based function that returns the expression string for
        the subtree rooted at `current`. We skip `parent` so as 
        not to go back up the tree.
        """
        children = [c for c in adjacency[current] if c != parent]

        # If no children, it's a leaf
        if len(children) == 0:
            return self._leaf_expr(node_info[current])

        # Otherwise, gather sub-expressions + our own numeric label
        subexprs = []
        for c in children:
            subexprs.append(self._build_expression_string(c, current, adjacency, node_info))

        # Also include current node's numeric label as an operand
        subexprs.append(str(node_info[current]["value"]))

        # shuffle them if operator is commutative
        # (for + or *, we consider them commutative)
        random.shuffle(subexprs)

        joined = (" " + node_info[current]["operator"] + " ").join(subexprs)

        if node_info[current]["sign"] == "-":
            return f"-({joined})"
        else:
            return f"({joined})"

    def _leaf_expr(self, node_data):
        """
        Returns string for a leaf node, e.g. "-5" or "7".
        """
        val_str = str(node_data["value"])
        if node_data["sign"] == "-":
            return "-" + val_str
        else:
            return val_str


def _evaluate_one_innermost(expr_str, mod = 10):
    """
    Finds all innermost bracketed sub-expressions in 'expr_str', collects them,
    and randomly chooses exactly one to evaluate and replace.
    If none exist, returns 'expr_str' unchanged.
    """
    stack = []
    innermost_positions = []

    for i, ch in enumerate(expr_str):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            start = stack.pop()
            sub = expr_str[start+1 : i]
            # If 'sub' doesn't contain '(' or ')', it's innermost
            if '(' not in sub and ')' not in sub:
                innermost_positions.append((start, i))

    # Randomly pick one innermost bracket
    if not innermost_positions:
        return expr_str
    start, end = random.choice(innermost_positions)
    sub_expr = expr_str[start+1 : end]

    val = eval(sub_expr)%mod
    val_str = str(val)

    # Replace that bracket with its evaluated value
    new_expr_str = expr_str[:start] + val_str + expr_str[end+1:]
    return new_expr_str

class RandomPruferExpressionDatasetCurriculum(Dataset):
    def __init__(
        self,
        size=1000,         # how many expressions to generate
        min_n=1,           # minimum number of nodes
        max_n=6,           # maximum number of nodes
        numbers=range(1, 10)  # possible numeric labels
    ):
        self.rdpf = RandomPruferExpressionDataset(size, min_n, max_n, numbers)
        for i in range(len(self.rdpf)):
            self.rdpf.labels[i] = _evaluate_one_innermost(self.rdpf.data[i])

    def __len__(self):
        return len(self.rdpf)

    def __getitem__(self, idx):
        return {
            "instruction": "Evaluate this expression modulo 10, try expanding one bracket.",
            "input": self.rdpf.data[idx],
            "output": self.rdpf.labels[idx],
        }


class ParityDataset(Dataset):
    def __init__(self, length, seq_len):
        self.length = length
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        for _ in range(length):
            len = random.randint(1, seq_len)
            binary_str = ''.join(str(torch.randint(2, size=()).item()) for _ in range(len)) #len
            #for _ in range(seq_len - len):
            #  binary_str = binary_str + "0"
            data = ""
            for i in range(len):
            #for i in range(seq_len):
              data = data + binary_str[i] + " "
            self.data.append(data)
            self.labels.append(sum(int(bit) for bit in binary_str) % 2)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'instruction' : "Output the parity of this sequence.", "input":self.data[idx], "output": str(self.labels[idx])}


class QAPairDataset(Dataset):
    """
    PyTorch-compatible dataset for lines that look like:
        <question text> || <any COT> #### <answer>
    """

    def __init__(self, filepath: str):
        self._data = []                     # list of (question, answer) tuples
        self.label = []
        with open(filepath, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue                # skip blanks

                # split on first "||"
                try:
                    question, rest = line.split("||", 1)
                except ValueError:
                    continue                # malformed → skip

                marker = "####"
                if marker not in rest:
                    continue                # no answer marker → skip

                answer = rest.rsplit(marker, 1)[1].strip()
                # self._data.append({"text": "Question:\n" + question + "\n" + "Answer:\n" + "#### " + answer + "\n" + "Question:\n"})
                self._data.append(question)
                self.label.append(answer)
    # ---------- required by torch.utils.data.Dataset ----------
    def __len__(self):
        """Number of examples in the dataset."""
        return len(self._data)

    def __getitem__(self, idx):
        """Return the (question, answer) pair at position `idx`."""
        return {'instruction' : "Answer the math question.", "input":self._data[idx], "output": self.label[idx]}

class QAPairDatasetWithCoT(Dataset):
    def __init__(self, filepath: str):
        self._data = []                     # list of (question, answer) tuples
        self.label = []
        with open(filepath, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue                # skip blanks

                # split on first "||"
                try:
                    question, rest = line.split("||", 1)
                except ValueError:
                    continue                # malformed → skip

                # self._data.append({"text": "Question:\n" + question + "\n" + "Answer:\n" + "#### " + answer + "\n" + "Question:\n"})
                self._data.append(question)
                self.label.append(rest)
    # ---------- required by torch.utils.data.Dataset ----------
    def __len__(self):
        """Number of examples in the dataset."""
        return len(self._data)

    def __getitem__(self, idx):
        """Return the (question, answer) pair at position `idx`."""
        return {'instruction' : "Answer the math question.", "input":self._data[idx], "output": self.label[idx]}

class ParityDatasetSimplified(Dataset):
    def __init__(self, length, seq_len):
        self.length = length
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        for _ in range(length):
            len = random.randint(1, seq_len)
            binary_str = ''.join(str(torch.randint(2, size=()).item()) for _ in range(len)) #len
            #for _ in range(seq_len - len):
            #  binary_str = binary_str + "0"
            data = ""
            for i in range(len):
            #for i in range(seq_len):
              data = data + binary_str[i] + " "
            self.data.append(data)
            self.labels.append(sum(int(bit) for bit in binary_str) % 2)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'instruction' : "Decide parity.", "input":self.data[idx], "output": str(self.labels[idx])}


class ParityDatasetFixedLength(Dataset):
    def __init__(self, length, seq_len):
        self.length = length
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        for _ in range(length):
            len = seq_len
            binary_str = ''.join(str(torch.randint(2, size=()).item()) for _ in range(len)) #len
            data = ""
            for i in range(len):
              data = data + binary_str[i] + " "
            self.data.append(data)
            self.labels.append(sum(int(bit) for bit in binary_str) % 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'instruction' : "Output the parity of this sequence.", "input":self.data[idx], "output": str(self.labels[idx])}
