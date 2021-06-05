
train_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]


def class_counts(rows):
    counts = {}

    for row in rows:
        label = row[-1]

        if label not in counts:
            counts[label] = 0

        counts[label] += 1
    return counts


print(class_counts(train_data))


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):

        val = example[self.column]
        if is_numeric(self.value):
            return val >= self.value

        else:
            return val == self.value

    def __repr__(self):
        condition = "=="

        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


def partition(rows, question):

    true_rows, false_rows = [], []

    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)

    return true_rows, false_rows


def gini(rows):

    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):

    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):

    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)

    n_features = len(rows[0]) - 1

    for col in range(n_features):

        values = set([row[col] for row in rows])

        for val in values:
            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


"""Test code"""
best_gain, best_q = find_best_split(train_data)
print(best_q)


class Leaf:

    def __init__(self, rows):
        self.pred = class_counts(rows)


class Node:

    def __init__(self, true_rows, false_rows, question):
        self.left_branch = true_rows
        self.right_branch = false_rows
        self.question = question


def build_tree(rows):

    best_gain, best_q = find_best_split(rows)

    if best_gain == 0: return Leaf(rows)

    true_rows, false_rows = partition(rows, best_q)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Node(true_branch, false_branch, best_q)


tree = build_tree(train_data)


def print_tree(tree):

    if isinstance(tree, Leaf):
        print("\tPredict", tree.pred)
        return

    print("\t", str(tree.question))

    print("\tTrue:")
    print_tree(tree.left_branch)
    print("\tFalse:")
    print_tree(tree.right_branch)


print_tree(tree)


